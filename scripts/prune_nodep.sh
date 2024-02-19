#!/bin/bash

SCRIPT_B='scripts/b.sh'
BASE_MODEL='baffo32/decapoda-research-llama-7B-hf'

task(){
  CUDA_VISIBLE_DEVICES=$2 python prune.py --pruning_ratio 0.25 --device cpu --eval_device cuda --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 30 --block_attention_layer_start 4 --block_attention_layer_end 30 --save_ckpt_log_name "$1" --pruner_type taylor --taylor "$3" --save_model
}
task2(){
  CUDA_VISIBLE_DEVICES=$2 python prune.py --pruning_ratio 0.2 --device cpu --eval_device cuda --block_wise --block_mlp_layer_start 0 --block_mlp_layer_end 32 --block_attention_layer_start 0 --block_attention_layer_end 32 --save_ckpt_log_name "$1" --pruner_type taylor --taylor "$3" --save_model
}

task 'llama_prune_0.2_without_tune_block_param_first_4_30_0.25_nodep_shape_index_grouping' 1 'param_first'& \
task 'llama_prune_0.2_without_tune_block_param_second_4_30_0.25_nodep_shape_index_grouping' 2 'param_second'& \
task 'llama_prune_0.2_without_tune_block_param_mix_4_30_0.25_nodep_shape_index_grouping' 3 'param_mix'& \
task 'llama_prune_0.2_without_tune_block_vectorize_4_30_0.25_nodep_shape_index_grouping' 0 'vectorize'

wait
CUDA_VISIBLE_DEVICES=1 bash $SCRIPT_B $BASE_MODEL prune_log/llama_prune_0.2_without_tune_block_param_first_4_30_0.25_nodep_shape_index_grouping&
CUDA_VISIBLE_DEVICES=2 bash $SCRIPT_B $BASE_MODEL prune_log/llama_prune_0.2_without_tune_block_param_second_4_30_0.25_nodep_shape_index_grouping &
CUDA_VISIBLE_DEVICES=3 bash $SCRIPT_B $BASE_MODEL prune_log/llama_prune_0.2_without_tune_block_param_mix_4_30_0.25_nodep_shape_index_grouping &
CUDA_VISIBLE_DEVICES=0 bash $SCRIPT_B $BASE_MODEL prune_log/llama_prune_0.2_without_tune_block_vectorize_4_30_0.25_nodep_shape_index_grouping
wait

echo "所有任务完成"