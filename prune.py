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
        self.bias_mask = nn.Parameter(torch.ones(self.bias.size()), requires_grad=False)

    def forward(self, input):
        masked_weight = self.weight * self.weight_mask
        masked_bias = self.bias * self.bias_mask
        output = F.linear(input, masked_weight, masked_bias)
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
            local_norm = local_norm[idxs]
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


def plot(figure, x, y):
    plt.figure(figsize=figure)
    for index in range(len(y)):
        plt.subplot((len(y) + 1) // 2, 2, index + 1)
        plt.title(y[index][0])
        plt.plot(x, y[index][1], label=y[index][0])
    plt.show()
    return


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
        # [1, 319, 11473, 2643, 378, 629, 271, 18099],
        # [1, 4103, 9632, 4223, 304, 5176, 29901, 13],
    ]).to(
        args.device)

    # forward_prompts = get_examples('bookcorpus', tokenizer, 10, seq_len=64).to(args.device)

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

    for i in range(args.iterative_steps):
        if pruner_type in ['taylor']:
            example_prompts = get_examples('bookcorpus', tokenizer, 10, seq_len=64).to(args.device)
            logger.log("Start Backwarding in iterative steps = {}...".format(i))
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

        def apply_mask(layer, idxs, prune_fn):
            idxs.sort(reverse=True)
            before = layer.weight.numel()
            if prune_fn in ["linear_out"]:
                rows_mask = torch.ones(layer.weight.data.size(0), dtype=torch.bool)
                rows_mask[idxs] = False
                layer.weight.data = layer.weight.data[rows_mask]
                layer.out_features -= len(idxs)
            elif prune_fn in ["linear_in"]:
                cols_mask = torch.ones(layer.weight.data.size(1), dtype=torch.bool)
                cols_mask[idxs] = False
                layer.weight.data = layer.weight.data[:, cols_mask]
                layer.in_features -= len(idxs)
            return 1 - layer.weight.numel() / before

        # pruner.step()
        if args.global_pruning:
            whole_imps_attn = torch.tensor([]).to(args.device)
            whole_imps_attn_scaled = torch.tensor([]).to(args.device)
            whole_imps_attn_scaled_layer = torch.tensor([]).to(args.device)

            whole_imps_mlp = torch.tensor([]).to(args.device)
            whole_imps_mlp_scaled = torch.tensor([]).to(args.device)
            whole_imps_mlp_scaled_layer = torch.tensor([]).to(args.device)

            pruning_ratio_mha = []
            pruning_ratio_mlp = []

            weight_norm_mha = []
            activation_norm_mha = []
            gradient_norm_mha = []
            importance_norm_mha = []
            importance_norm_mha_scale = []

            weight_norm_mlp = []
            activation_norm_mlp = []
            gradient_norm_mlp = []
            importance_norm_mlp = []
            importance_norm_mlp_scale = []

            layer_x_mha = [x for x in range(args.block_attention_layer_start, args.block_attention_layer_end)]
            layer_x_mlp = [x for x in range(args.block_mlp_layer_start, args.block_mlp_layer_end)]

            for z in range(args.block_attention_layer_start, args.block_attention_layer_end):
                layer = model.model.layers[z]
                weight_norm_mha.append(torch.linalg.matrix_norm(layer.self_attn.q_proj.weight).tolist())
                gradient_norm_mha.append(torch.linalg.matrix_norm(layer.self_attn.q_proj.weight.grad).tolist())
                imps = imp(layer.self_attn.q_proj, "linear_out", [1])
                importance_norm_mha.append(torch.linalg.vector_norm(imps).tolist())
                imps_scaled = imps.clone()
                mean = torch.mean(imps_scaled)
                stdev = torch.std(imps_scaled, unbiased=False)
                imps_scaled = (imps_scaled - mean) / stdev
                importance_norm_mha_scale.append(torch.linalg.vector_norm(imps_scaled).tolist())
                imps_scaled = imps_scaled.view(-1, layer.self_attn.head_dim).sum(1)
                whole_imps_attn_scaled_layer = torch.cat((whole_imps_attn_scaled_layer, imps_scaled), dim=0)
                imps = imps.view(-1, layer.self_attn.head_dim).sum(1)
                whole_imps_attn = torch.cat((whole_imps_attn, imps), dim=0)

            for z in range(args.block_mlp_layer_start, args.block_mlp_layer_end):
                layer = model.model.layers[z]
                weight_norm_mlp.append(torch.linalg.matrix_norm(layer.mlp.gate_proj.weight).tolist())
                gradient_norm_mlp.append(torch.linalg.matrix_norm(layer.mlp.gate_proj.weight.grad).tolist())
                imps = imp(layer.mlp.gate_proj, "linear_out", [1])
                importance_norm_mlp.append(torch.linalg.vector_norm(imps).tolist())
                imps_scaled = imps.clone()
                mean = torch.mean(imps_scaled)
                stdev = torch.std(imps_scaled, unbiased=False)
                imps_scaled = (imps_scaled - mean) / stdev
                importance_norm_mlp_scale.append(torch.linalg.vector_norm(imps_scaled).tolist())
                whole_imps_mlp_scaled_layer = torch.cat((whole_imps_mlp_scaled_layer, imps_scaled), dim=0)
                whole_imps_mlp = torch.cat((whole_imps_mlp, imps), dim=0)

            mean = torch.mean(whole_imps_attn)
            stdev = torch.std(whole_imps_attn, unbiased=False)
            whole_imps_attn_scaled = (whole_imps_attn - mean) / stdev

            plot((15, 10), [x for x in range(len(whole_imps_attn))],
                 [("No scale Importance MHA", whole_imps_attn.tolist()),
                  ("Scaled Importance global-wise", whole_imps_attn_scaled.tolist()),
                  ("Scaled Importance layer-wise", whole_imps_attn_scaled_layer.tolist())])

            mean = torch.mean(whole_imps_mlp)
            stdev = torch.std(whole_imps_mlp, unbiased=False)
            whole_imps_mlp_scaled = (whole_imps_mlp - mean) / stdev

            plot((15, 10), [x for x in range(len(whole_imps_mlp))],
                 [("No scale Importance MLP", whole_imps_mlp.tolist()),
                  ("Scaled Importance global-wise", whole_imps_mlp_scaled.tolist()),
                  ("Scaled Importance layer-wise", whole_imps_mlp_scaled_layer.tolist())])

            # imp_argsort = torch.argsort(whole_imps_attn_scaled_layer)
            imp_argsort = torch.argsort(whole_imps_attn_scaled_layer)
            n_pruned = len(imp_argsort) - int(
                len(imp_argsort) *
                (1 - args.pruning_ratio)
            )
            pruning_groups = imp_argsort[:n_pruned]
            pruning_groups = pruning_groups.tolist()
            pruning_groups.sort()
            for z in range(args.block_attention_layer_start, args.block_attention_layer_end):
                layer = model.model.layers[z]
                pruning_idxs = torch.tensor([], dtype=torch.int8)
                for j in range(layer.self_attn.num_heads):
                    # i-> current layer index
                    # j-> current head index (inside current layer)
                    if (z - args.block_attention_layer_start) * layer.self_attn.num_heads + j in pruning_groups:
                        pruning_idxs = torch.cat(
                            (pruning_idxs,
                             torch.tensor([j * layer.self_attn.head_dim + x for x in range(layer.self_attn.head_dim)])
                             ),
                            dim=0)
                pruning_ratio_mha.append(apply_mask(layer.self_attn.q_proj, pruning_idxs.tolist(), "linear_out"))
                apply_mask(layer.self_attn.k_proj, pruning_idxs.tolist(), "linear_out")
                apply_mask(layer.self_attn.v_proj, pruning_idxs.tolist(), "linear_out")
                apply_mask(layer.self_attn.o_proj, pruning_idxs.tolist(), "linear_in")

            plot((18, 6), layer_x_mha,
                 [('mha', pruning_ratio_mha), ('gradient', gradient_norm_mha), ('weight', weight_norm_mha),
                  ('tylor imp', importance_norm_mha), ('scaled imp', importance_norm_mha_scale),
                  ])
            # for z in range(args.block_mlp_layer_start, args.block_mlp_layer_end):
            #     layer = model.model.layers[z]
            #     imps = imp(layer.mlp.gate_proj, "linear_out", [])
            #     pruning_idxs = get_mask(imps, layer.mlp.gate_proj, "linear_out",
            #                             args.pruning_ratio)
            #     apply_mask(layer.mlp.gate_proj, pruning_idxs.tolist(), "linear_out")
            #     apply_mask(layer.mlp.up_proj, pruning_idxs.tolist(), "linear_out")
            #     apply_mask(layer.mlp.down_proj, pruning_idxs.tolist(), "linear_in")
            imp_argsort = torch.argsort(whole_imps_mlp_scaled_layer)
            n_pruned = len(imp_argsort) - int(
                len(imp_argsort) *
                (1 - args.pruning_ratio)
            )
            pruning_groups = imp_argsort[:n_pruned]
            pruning_groups = pruning_groups.tolist()
            pruning_groups.sort()

            for z in range(args.block_mlp_layer_start, args.block_mlp_layer_end):
                layer = model.model.layers[z]
                pruning_idxs = torch.tensor([], dtype=torch.int8)
                for j in range(layer.mlp.gate_proj.out_features):
                    # z-> current layer index
                    # j-> current vector index (inside current layer)
                    if (z - args.block_attention_layer_start) * layer.mlp.gate_proj.out_features + j in pruning_groups:
                        pruning_idxs = torch.cat(
                            (pruning_idxs,
                             torch.tensor([j])
                             ),
                            dim=0)
                pruning_ratio_mlp.append(apply_mask(layer.mlp.gate_proj, pruning_idxs.tolist(), "linear_out"))
                apply_mask(layer.mlp.up_proj, pruning_idxs.tolist(), "linear_out")
                apply_mask(layer.mlp.down_proj, pruning_idxs.tolist(), "linear_in")

            plot((18, 6), layer_x_mlp,
                 [('mlp', pruning_ratio_mlp), ('gradient', gradient_norm_mlp), ('weight', weight_norm_mlp),
                  ('tylor imp', importance_norm_mlp), ('scaled imp', importance_norm_mlp_scale),
                  ])
        else:

            for z in range(args.block_attention_layer_start, args.block_attention_layer_end):
                layer = model.model.layers[z]
                if args.mask_type_mha == 'q':
                    imps = imp(layer.self_attn.q_proj, "linear_out", [])
                    pruning_idxs = get_mask(imps, layer.self_attn.q_proj, "linear_out",
                                            args.pruning_ratio,
                                            layer.self_attn.head_dim if args.head_dim else 1)
                elif args.mask_type_mha == 'k':
                    imps = imp(layer.self_attn.k_proj, "linear_out", [])
                    pruning_idxs = get_mask(imps, layer.self_attn.k_proj, "linear_out",
                                            args.pruning_ratio,
                                            layer.self_attn.head_dim if args.head_dim else 1)
                elif args.mask_type_mha == 'v':
                    imps = imp(layer.self_attn.v_proj, "linear_out", [])
                    pruning_idxs = get_mask(imps, layer.self_attn.v_proj, "linear_out",
                                            args.pruning_ratio,
                                            layer.self_attn.head_dim if args.head_dim else 1)
                elif args.mask_type_mha == 'o':
                    imps = imp(layer.self_attn.o_proj, "linear_in", [])
                    pruning_idxs = get_mask(imps, layer.self_attn.o_proj, "linear_in",
                                            args.pruning_ratio,
                                            layer.self_attn.head_dim if args.head_dim else 1)
                else:
                    raise NotImplementedError()
                apply_mask(layer.self_attn.q_proj, pruning_idxs.tolist(), "linear_out")
                apply_mask(layer.self_attn.k_proj, pruning_idxs.tolist(), "linear_out")
                apply_mask(layer.self_attn.v_proj, pruning_idxs.tolist(), "linear_out")
                apply_mask(layer.self_attn.o_proj, pruning_idxs.tolist(), "linear_in")

            for z in range(args.block_mlp_layer_start, args.block_mlp_layer_end):
                layer = model.model.layers[z]
                if args.mask_type_mlp == 'gate_proj':
                    imps = imp(layer.mlp.gate_proj, "linear_out", [])
                    pruning_idxs = get_mask(imps, layer.mlp.gate_proj, "linear_out",
                                            args.pruning_ratio)
                elif args.mask_type_mlp == 'up_proj':
                    imps = imp(layer.mlp.up_proj, "linear_out", [])
                    pruning_idxs = get_mask(imps, layer.mlp.up_proj, "linear_out",
                                            args.pruning_ratio)
                elif args.mask_type_mlp == 'down_proj':
                    imps = imp(layer.mlp.down_proj, "linear_in", [])
                    pruning_idxs = get_mask(imps, layer.mlp.down_proj, "linear_in",
                                            args.pruning_ratio)
                else:
                    raise NotImplementedError()
                apply_mask(layer.mlp.gate_proj, pruning_idxs.tolist(), "linear_out")
                apply_mask(layer.mlp.up_proj, pruning_idxs.tolist(), "linear_out")
                apply_mask(layer.mlp.down_proj, pruning_idxs.tolist(), "linear_in")
    after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    for layer in model.model.layers:
        layer.self_attn.num_heads = layer.self_attn.q_proj.weight.data.shape[0] // layer.self_attn.head_dim
    logger.log("#Param before: {}, #Param after: {}, Ratio = {:.4f}%".format(before_pruning_parameters,
                                                                             after_pruning_parameters,
                                                                             100.0 * after_pruning_parameters / before_pruning_parameters))

    gc.collect()
    torch.cuda.empty_cache()

    if args.save_model:
        model.half()
        torch.save({
            'model': model,
            'tokenizer': tokenizer,
        }, logger.best_checkpoint_path)

    if args.eval_device != "cpu":
        model.half()
    model.to(args.eval_device)

    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if args.test_after_train:
        logger.log("\n==================Generation Results After Pruning================\n")

        model.eval()
        with torch.no_grad():
            for prompt in prompts:
                input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to(args.eval_device)

                generation_output = model.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    top_k=50,
                    max_length=args.max_seq_len,
                    top_p=args.top_p,
                    temperature=args.temperature,
                )

                result = tokenizer.decode(generation_output[0])
                logger.log(result)

        logger.log("\n==================Finish================\n")

    ppl = PPLMetric(model, tokenizer, ['wikitext2', 'ptb'], args.max_seq_len, device=args.eval_device)
    logger.log("PPL after pruning: {}".format(ppl))
    logger.log("Memory Requirement: {} MiB\n".format(torch.cuda.memory_allocated() / 1024 / 1024))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pruning LLaMA (huggingface version)')

    # argument for parsing
    parser.add_argument('--base_model', type=str, default="baffo32/decapoda-research-llama-7B-hf",
                        help='base model name')
    parser.add_argument('--save_ckpt_log_name', type=str, default="llama_prune",
                        help='the path for save the checkpoint and the log. The final path would be log/{your_name_here}_{pruner_type}_{pruning_ratio}')
    parser.add_argument('--pruning_ratio', type=float, default=0.5, help='pruning ratio')
    parser.add_argument('--pruner_type', type=str, default='taylor', help='pruner type')
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
