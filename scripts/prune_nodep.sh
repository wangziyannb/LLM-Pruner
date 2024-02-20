#!/bin/bash
SCRIPT_A='scripts/a.sh'
SCRIPT_B='scripts/b.sh'
BASE_MODEL='baffo32/decapoda-research-llama-7B-hf'
tune0='llama_prune_0.2_with_tune_block_param_first_4_30_0.25_nodep_shape_index_grouping'
tune1='llama_prune_0.2_with_tune_block_param_second_4_30_0.25_nodep_shape_index_grouping'
tune2='llama_prune_0.2_with_tune_block_param_mix_4_30_0.25_nodep_shape_index_grouping'
tune3='llama_prune_0.2_with_tune_block_vectorize_4_30_0.25_nodep_shape_index_grouping'

prune0='llama_prune_0.2_without_tune_block_param_first_4_30_0.25_nodep_shape_index_grouping'
prune1='llama_prune_0.2_without_tune_block_param_second_4_30_0.25_nodep_shape_index_grouping'
prune2='llama_prune_0.2_without_tune_block_param_mix_4_30_0.25_nodep_shape_index_grouping'
prune3='llama_prune_0.2_without_tune_block_vectorize_4_30_0.25_nodep_shape_index_grouping'

task_skip(){
  CUDA_VISIBLE_DEVICES=$2 python prune.py --pruning_ratio 0.25 --device cpu --eval_device cuda --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 30 --block_attention_layer_start 4 --block_attention_layer_end 30 --save_ckpt_log_name "$1" --pruner_type taylor --taylor "$3" --save_model
}
task_noskip(){
  CUDA_VISIBLE_DEVICES=$2 python prune.py --pruning_ratio 0.2 --device cpu --eval_device cuda --block_wise --block_mlp_layer_start 0 --block_mlp_layer_end 32 --block_attention_layer_start 0 --block_attention_layer_end 32 --save_ckpt_log_name "$1" --pruner_type taylor --taylor "$3" --save_model
}
task_post_training(){
  CUDA_VISIBLE_DEVICES=$3 python post_training.py --prune_model prune_log/"$1"/pytorch_model.bin --data_path yahma/alpaca-cleaned --output_dir tune_log/"$2" --wandb_project "$2" --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64
}
task_noskip "$prune0" 1 'param_first'& \
task_noskip "$prune1" 2 'param_second'& \
task_noskip "$prune2" 3 'param_mix'& \
task_noskip "$prune3" 0 'vectorize'

wait
CUDA_VISIBLE_DEVICES=1 bash $SCRIPT_B $BASE_MODEL prune_log/"$prune0" &
CUDA_VISIBLE_DEVICES=2 bash $SCRIPT_B $BASE_MODEL prune_log/"$prune1" &
CUDA_VISIBLE_DEVICES=3 bash $SCRIPT_B $BASE_MODEL prune_log/"$prune2" &
CUDA_VISIBLE_DEVICES=0 bash $SCRIPT_B $BASE_MODEL prune_log/"$prune3"
wait

task_post_training "$prune0" "$tune0" 1 &
task_post_training "$prune1" "$tune1" 2 &
task_post_training "$prune2" "$tune2" 3 &
task_post_training "$prune3" "$tune3" 0

wait

CUDA_VISIBLE_DEVICES=1 bash $SCRIPT_A $BASE_MODEL tune_log/"$tune0" prune_log/"$prune0" 1400 &
CUDA_VISIBLE_DEVICES=2 bash $SCRIPT_A $BASE_MODEL tune_log/"$tune1" prune_log/"$prune1" 1400 &
CUDA_VISIBLE_DEVICES=3 bash $SCRIPT_A $BASE_MODEL tune_log/"$tune2" prune_log/"$prune2" 1400 &
CUDA_VISIBLE_DEVICES=0 bash $SCRIPT_A $BASE_MODEL tune_log/"$tune3" prune_log/"$prune3" 1400

wait

CUDA_VISIBLE_DEVICES=1 python generate.py --model_type tune_prune_LLM --ckpt prune_log/"$prune0"/pytorch_model.bin --lora_ckpt tune_log/"$tune0"/checkpoint-1400 --base_model "$BASE_MODEL" &
CUDA_VISIBLE_DEVICES=1 python generate.py --model_type tune_prune_LLM --ckpt prune_log/"$prune1"/pytorch_model.bin --lora_ckpt tune_log/"$tune1"/checkpoint-1400 --base_model "$BASE_MODEL" &
CUDA_VISIBLE_DEVICES=3 python generate.py --model_type tune_prune_LLM --ckpt prune_log/"$prune2"/pytorch_model.bin --lora_ckpt tune_log/"$tune2"/checkpoint-1400 --base_model "$BASE_MODEL" &
CUDA_VISIBLE_DEVICES=0 python generate.py --model_type tune_prune_LLM --ckpt prune_log/"$prune3"/pytorch_model.bin --lora_ckpt tune_log/"$tune3"/checkpoint-1400 --base_model "$BASE_MODEL"
wait
echo "所有任务完成"