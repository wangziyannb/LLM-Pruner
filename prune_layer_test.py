import os
import gc
import sys
import time
import json
import copy
import random
import argparse
from typing import Tuple

import torch
import numpy as np
from torch import Tensor
from transformers import LlamaTokenizer, GenerationConfig, LlamaConfig
from LLMPruner.models.hf_llama.modeling_llama import LlamaForCausalLM, LlamaRMSNorm, LlamaAttention, LlamaMLP

import LLMPruner.torch_pruning as tp
from LLMPruner.pruner import hf_llama_pruner as llama_pruner
from LLMPruner.utils.logger import LoggerWithDepth
from LLMPruner.evaluator.ppl import PPLMetric
from LLMPruner.datasets.example_samples import get_examples
from LLMPruner.templates.prompts import prompts

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import csv
import pandas as pd


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.weight_mask = nn.Parameter(torch.ones(self.weight.size()), requires_grad=False)

    def forward(self, input):
        masked_weight = self.weight * self.weight_mask
        output = F.linear(input, masked_weight)
        # for i in masked_weight:
        #     print(i)
        return output

    def apply_mask(self, idxs, prune_fn):
        # self.weight_mask = nn.Parameter(torch.zeros(self.weight.size()), requires_grad=False)
        pruned_params = 0
        if prune_fn in ["linear_out"]:
            for i in idxs:
                self.weight_mask[i, :] = 0
                pruned_params += len(self.weight_mask[i])
        elif prune_fn in ["linear_in"]:
            for i in idxs:
                self.weight_mask[:, i] = 0
                pruned_params += len(self.weight_mask)
        return pruned_params

    def reset_mask(self):
        self.weight_mask = nn.Parameter(torch.ones(self.weight.size()), requires_grad=False)


class TaylorImportance(tp.importance.Importance):
    def __init__(self, group_reduction="sum", normalizer=None, taylor=None):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.taylor = taylor

    def _reduce(self, group_imp):
        if self.group_reduction == "sum":
            group_imp = group_imp.sum(dim=0)
        elif self.group_reduction == "mean":
            group_imp = group_imp.mean(dim=0)
        elif self.group_reduction == "max":
            group_imp = group_imp.max(dim=0)[0]
        elif self.group_reduction == "prod":
            group_imp = torch.prod(group_imp, dim=0)
        elif self.group_reduction == 'first':
            group_imp = group_imp[0]
        elif self.group_reduction == 'second':
            group_imp = group_imp[1]
        elif self.group_reduction is None:
            group_imp = group_imp
        else:
            raise NotImplementedError
        return group_imp

    @torch.no_grad()
    def __call__(self, layer, prune_fn, idxs):

        group_imp = []

        idxs.sort()

        if prune_fn not in [
            "linear_out", "linear_in", "embedding_out", "rmsnorm_out"
        ]:
            return

        salience = layer.weight * layer.weight.grad

        if self.taylor in ['param_second']:
            salience = layer.weight * layer.weight.acc_grad * layer.weight
        elif self.taylor in ['param_mix']:
            salience = salience - 0.5 * layer.weight * layer.weight.acc_grad * layer.weight

        # Linear out_channels
        if prune_fn in ["linear_out"]:
            if self.taylor == 'vectorize':
                local_norm = salience.sum(1).abs()
            elif 'param' in self.taylor:
                local_norm = salience.abs().sum(1)
            else:
                raise NotImplementedError
            group_imp.append(local_norm)

        # Linear in_channels
        elif prune_fn in ["linear_in"]:
            if self.taylor == 'vectorize':
                local_norm = salience.sum(0).abs()
            elif 'param' in self.taylor:
                local_norm = salience.abs().sum(0)
            else:
                raise NotImplementedError
            group_imp.append(local_norm)

        # RMSNorm
        elif prune_fn == "rmsnorm_out":
            local_norm = salience.abs()
            group_imp.append(local_norm)

        # Embedding
        elif prune_fn == "embedding_out":
            if self.taylor == 'vectorize':
                local_norm = salience[:, idxs].sum(0).abs()
            elif 'param' in self.taylor:
                local_norm = salience[:, idxs].abs().sum(0)
            else:
                raise NotImplementedError
            group_imp.append(local_norm)

        if len(group_imp) == 0:
            return None

        min_imp_size = min([len(imp) for imp in group_imp])
        aligned_group_imp = []
        for imp in group_imp:
            if len(imp) > min_imp_size and len(imp) % min_imp_size == 0:
                imp = imp.view(len(imp) // min_imp_size, min_imp_size).sum(0)
                aligned_group_imp.append(imp)
            elif len(imp) == min_imp_size:
                aligned_group_imp.append(imp)
        group_imp = torch.stack(aligned_group_imp, dim=0)
        group_imp = self._reduce(group_imp)
        # if self.normalizer is not None:
        # group_imp = self.normalizer(group, group_imp)
        return group_imp


class RandomImportance(tp.importance.Importance):
    @torch.no_grad()
    def __call__(self, layer, prune_fn, idxs):
        return torch.rand(len(idxs))


def get_mask(imps, layer, prune_fn, target_sparsity, head_dim=1):
    if prune_fn in ["linear_out"]:
        current_channels = layer.out_features
    elif prune_fn in ["linear_in"]:
        current_channels = layer.in_features
    else:
        current_channels = layer.out_features
    n_pruned = current_channels - int(
        current_channels *
        (1 - target_sparsity)
    )
    if n_pruned <= 0:
        return

    if head_dim > 1:
        imps = imps.view(-1, head_dim).sum(1)

    imp_argsort = torch.argsort(imps)

    if head_dim > 1:
        # n_pruned//consecutive_groups
        pruning_groups = imp_argsort[:(n_pruned // head_dim)]
        group_size = head_dim
        pruning_idxs = torch.cat(
            [torch.tensor([j + group_size * i for j in range(group_size)])
             for i in pruning_groups], 0)
    else:
        pruning_idxs = imp_argsort[:n_pruned]
    # print(len(pruning_idxs))
    return pruning_idxs


def main(args):
    logger = LoggerWithDepth(
        env_name="{}".format(args.save_ckpt_log_name),
        config=args.__dict__,
        root_dir='prune_log',
        setup_sublogger=True
    )
    set_random_seed(args.seed)
    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        low_cpu_mem_usage=True if args.torch_version >= 1.9 else False
    )
    if args.device != "cpu":
        model.half()
    model.to(args.device)
    pruner_type = args.pruner_type.lower()
    assert pruner_type in ['random', 'l2', 'l1', 'taylor']
    for param in model.parameters():
        # if param.requires_grad:
        #     print(param.numel())
        param.requires_grad_(True)
    before_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    forward_prompts = torch.tensor([
        [1, 306, 4658, 278, 6593, 310, 2834, 338],
        [1, 3439, 17632, 1925, 29892, 278, 6368, 310],
    ]).to(
        args.device)

    if pruner_type == 'random':
        imp = RandomImportance()
    elif pruner_type == 'l1':
        imp = llama_pruner.MagnitudeImportance(p=1)
    elif pruner_type == 'l2':
        imp = llama_pruner.MagnitudeImportance(p=2)
    elif pruner_type == 'taylor':
        imp = TaylorImportance(group_reduction=args.grouping_strategy, taylor=args.taylor)
    else:
        raise NotImplementedError
    for i in model.model.layers:
        g = i.self_attn.q_proj.weight
        i.self_attn.q_proj = MaskedLinear(i.self_attn.q_proj.in_features, i.self_attn.q_proj.out_features, bias=False)
        i.self_attn.q_proj.weight = g

        g = i.self_attn.k_proj.weight
        i.self_attn.k_proj = MaskedLinear(i.self_attn.k_proj.in_features, i.self_attn.k_proj.out_features, bias=False)
        i.self_attn.k_proj.weight = g

        g = i.self_attn.v_proj.weight
        i.self_attn.v_proj = MaskedLinear(i.self_attn.v_proj.in_features, i.self_attn.v_proj.out_features, bias=False)
        i.self_attn.v_proj.weight = g

        g = i.self_attn.o_proj.weight
        i.self_attn.o_proj = MaskedLinear(i.self_attn.o_proj.in_features, i.self_attn.o_proj.out_features, bias=False)
        i.self_attn.o_proj.weight = g

        g = i.mlp.down_proj.weight
        i.mlp.down_proj = MaskedLinear(i.mlp.down_proj.in_features, i.mlp.down_proj.out_features, bias=False)
        i.mlp.down_proj.weight = g

        g = i.mlp.gate_proj.weight
        i.mlp.gate_proj = MaskedLinear(i.mlp.gate_proj.in_features, i.mlp.gate_proj.out_features, bias=False)
        i.mlp.gate_proj.weight = g

        g = i.mlp.up_proj.weight
        i.mlp.up_proj = MaskedLinear(i.mlp.up_proj.in_features, i.mlp.up_proj.out_features, bias=False)
        i.mlp.up_proj.weight = g

    logger.log("Use {} pruner...".format(pruner_type))
    logger.log("Pruning Attention Layer = {}".format(
        list(range(args.block_attention_layer_start, args.block_attention_layer_end))))
    logger.log("Pruning MLP Layer = {}".format(list(range(args.block_mlp_layer_start, args.block_mlp_layer_end))))
    logger.log("Start Pruning")

    model.eval()
    try:
        out = model(*forward_prompts)
    except:
        out = model(forward_prompts)

    for x in range(args.iterative_steps):
        if pruner_type in ['taylor']:
            example_prompts = get_examples('bookcorpus', tokenizer, 10, seq_len=64)
            logger.log("Start Backwarding in iterative steps = {}...".format(x))
            if args.taylor in ['param_mix', 'param_second']:
                for j in range(args.num_examples):
                    batch_input = example_prompts[j].unsqueeze(0)
                    loss = model(batch_input, labels=batch_input).loss
                    logger.log("Loss = {}".format(loss))
                    loss.backward()

                    for module_param in model.parameters():
                        if module_param.requires_grad:
                            module_param.grad = module_param.grad * module_param.grad / args.num_examples
                            if hasattr(module_param, 'acc_grad'):
                                module_param.acc_grad += module_param.grad
                            else:
                                module_param.acc_grad = copy.deepcopy(module_param.grad)
                    model.zero_grad()
                    del loss.grad
            loss = model(example_prompts, labels=example_prompts).loss
            logger.log("Loss = {}".format(loss))
            loss.backward()
        q = []
        k = []
        v = []
        o = []
        gate = []
        up = []
        down = []
        layer_index = [i for i in range(args.block_attention_layer_start, args.block_attention_layer_end)]
        layer_index_mlp = [i for i in range(args.block_mlp_layer_start, args.block_mlp_layer_end)]

        for i in range(args.block_attention_layer_start, args.block_attention_layer_end):
            layer = model.model.layers[i]
            layers_dict = {
                layer.self_attn.q_proj: q,
                layer.self_attn.k_proj: k,
                layer.self_attn.v_proj: v,
                layer.self_attn.o_proj: o,
            }
            for name, obj in layers_dict.items():
                if name in [layer.self_attn.o_proj]:
                    prune_fn = "linear_in"
                else:
                    prune_fn = "linear_out"

                imps = imp(name, prune_fn, [])
                pruning_idxs = get_mask(imps, name, prune_fn,
                                        args.pruning_ratio, layer.self_attn.head_dim)
                name.apply_mask(pruning_idxs.tolist(), prune_fn)

                new_model = model
                if args.eval_device != "cpu":
                    new_model.half()
                new_model.to(args.eval_device)

                new_model.config.pad_token_id = tokenizer.pad_token_id = 0
                new_model.config.bos_token_id = 1
                new_model.config.eos_token_id = 2

                ppl = PPLMetric(new_model, tokenizer, ['wikitext2'], args.max_seq_len, device=args.eval_device)
                logger.log("PPL after pruning: {}".format(ppl))
                obj.append(ppl["wikitext2"])
                gc.collect()
                torch.cuda.empty_cache()
                name.reset_mask()

        for i in range(args.block_mlp_layer_start, args.block_mlp_layer_end):
            layer = model.model.layers[i]
            layers_dict = {
                layer.mlp.gate_proj: gate,
                layer.mlp.up_proj: up,
                layer.mlp.down_proj: down,
            }
            for name, obj in layers_dict.items():
                if name in [layer.mlp.down_proj]:
                    prune_fn = "linear_in"
                else:
                    prune_fn = "linear_out"
                imps = imp(name, prune_fn, [])
                pruning_idxs = get_mask(imps, name, prune_fn,
                                        args.pruning_ratio, layer.self_attn.head_dim)
                name.apply_mask(pruning_idxs.tolist(), prune_fn)

                new_model = model
                if args.eval_device != "cpu":
                    new_model.half()
                new_model.to(args.eval_device)

                new_model.config.pad_token_id = tokenizer.pad_token_id = 0
                new_model.config.bos_token_id = 1
                new_model.config.eos_token_id = 2

                ppl = PPLMetric(new_model, tokenizer, ['wikitext2'], args.max_seq_len, device=args.eval_device)
                logger.log("PPL after pruning: {}".format(ppl))
                obj.append(ppl["wikitext2"])
                gc.collect()
                torch.cuda.empty_cache()
                name.reset_mask()

        csv_file = args.record_file_path
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Layer", "K", "Q", "V", "O", "Gate", "Up", "Down"])  # 写入标题
            for val1, val2, val3, val4, val5, val6, val7, val8 in zip(layer_index, k, q, v, o, gate, up, down):
                writer.writerow([val1, val2, val3, val4, val5, val6, val7, val8])
        # after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # logger.log("#Param before: {}, #Param after: {}, Ratio = {:.4f}%".format(before_pruning_parameters,
        #                                                                          after_pruning_parameters,
        #                                                                          100.0 * after_pruning_parameters / before_pruning_parameters))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pruning LLaMA (huggingface version)')

    # argument for parsing
    parser.add_argument('--base_model', type=str, default="baffo32/decapoda-research-llama-7B-hf",
                        help='base model name')
    parser.add_argument('--save_ckpt_log_name', type=str, default="llama_prune",
                        help='the path for save the checkpoint and the log. The final path would be log/{your_name_here}_{pruner_type}_{pruning_ratio}')
    parser.add_argument('--pruning_ratio', type=float, default=0.5, help='pruning ratio')
    parser.add_argument('--pruner_type', type=str, default='taylor', help='pruner type')
    parser.add_argument('--record_file_path', type=str, default='layer_sensitivity_0.2.csv', help='')
    parser.add_argument('--mask_type_mha', type=str, default='q', help='use what layer as mask (in attention)')
    parser.add_argument('--mask_type_mlp', type=str, default='gate_proj', help='use what layer as mask (in MLP)')
    parser.add_argument('--head_dim', action='store_true', help='use head dimention when pruning MHA')
    # argument for generation
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='top p')
    parser.add_argument('--max_seq_len', type=int, default=128, help='max sequence length')

    # argument for layer-wise pruning/column-wise pruning
    parser.add_argument('--block_wise', action='store_true', help='block wise')
    parser.add_argument('--layer_wise', action='store_true', help='layer wise')
    parser.add_argument('--channel_wise', action='store_true', help='channel wise')
    parser.add_argument('--layer', type=int, default=12, help='remain the previous n layers')

    parser.add_argument('--block_attention_layer_start', type=int, help='start layer of block attention layers',
                        default=3)
    parser.add_argument('--block_attention_layer_end', type=int, help='end layer of block attention layers', default=31)
    parser.add_argument('--block_mlp_layer_start', type=int, help='start layer of block mlp layers', default=3)
    parser.add_argument('--block_mlp_layer_end', type=int, help='end layer of block mlp layers', default=31)

    parser.add_argument('--iterative_steps', type=int, default=1, help="Iteration step for pruning. Default=1")
    parser.add_argument('--grouping_strategy', type=str, default='sum', help='Reduce method for grouping')
    parser.add_argument('--global_pruning', action='store_true', help='whether global pruning')
    parser.add_argument('--taylor', type=str, default='param_first',
                        help='choose from [vectorize, param_second, param_first, param_mix]')
    parser.add_argument('--num_examples', type=int, default=10)

    # general argument
    parser.add_argument('--device', type=str, default="cpu", help='device')
    parser.add_argument('--test_before_train', action='store_true', help='whether test before train')
    parser.add_argument('--eval_device', type=str, default="cuda", help='eval device')
    parser.add_argument('--test_after_train', action='store_true', help='whether test after train')

    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--save_model', action='store_true', help='if save model')
    args = parser.parse_args()

    torch_version = float('.'.join(torch.__version__.split('.')[:2]))
    args.torch_version = torch_version
    main(args)
