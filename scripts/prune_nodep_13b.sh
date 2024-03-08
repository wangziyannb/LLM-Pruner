#!/bin/bash
SCRIPT_A='scripts/a.sh'
SCRIPT_B='scripts/b.sh'
BASE_MODEL='srikanthmalla/decapoda-research-llama-13b-hf'
#tune0='llama_13b_prune_0.2_with_tune_block_param_first_4_30_0.25_nodep_shape_index_grouping'
#tune1='llama_13b_prune_0.2_with_tune_block_random_4_30_0.25_nodep_shape_index_grouping_HD'
tune2='llama_13b_prune_0.2_with_tune_block_param_first_0_40_0.2_nodep_shape_index_grouping'
tune3='llama_13b_prune_0.2_with_tune_block_random_0_40_0.2_nodep_shape_index_grouping_HD'

#prune0='llama_13b_prune_0.2_without_tune_block_param_first_4_30_0.25_nodep_shape_index_grouping'
#prune1='llama_13b_prune_0.2_without_tune_block_random_4_30_0.25_nodep_shape_index_grouping_HD'
prune2='llama_13b_prune_0.2_without_tune_block_param_first_0_40_0.2_nodep_shape_index_grouping'
prune3='llama_13b_prune_0.2_without_tune_block_random_0_40_0.2_nodep_shape_index_grouping_HD'

task_skip(){
  CUDA_VISIBLE_DEVICES=$2 python prune.py --base_model "$4" --pruning_ratio 0.24 --device cpu --eval_device cuda --head_dim --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 38 --block_attention_layer_start 4 --block_attention_layer_end 38 --save_ckpt_log_name "$1" --pruner_type taylor --taylor "$3" --save_model
}
task_noskip(){
  CUDA_VISIBLE_DEVICES=$2 python prune.py --base_model "$4" --pruning_ratio 0.2 --device cpu --eval_device cuda --head_dim --block_wise --block_mlp_layer_start 0 --block_mlp_layer_end 40 --block_attention_layer_start 0 --block_attention_layer_end 40 --save_ckpt_log_name "$1" --pruner_type taylor --taylor "$3" --save_model
}
task_skip_random_HD(){
  CUDA_VISIBLE_DEVICES=$2 python prune.py --base_model "$3" --pruning_ratio 0.24 --device cpu --eval_device cuda --head_dim --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 38 --block_attention_layer_start 4 --block_attention_layer_end 38 --save_ckpt_log_name "$1" --pruner_type random --save_model
}
task_noskip_random_HD(){
  CUDA_VISIBLE_DEVICES=$2 python prune.py --base_model "$3" --pruning_ratio 0.2 --device cpu --eval_device cuda --head_dim --block_wise --block_mlp_layer_start 0 --block_mlp_layer_end 40 --block_attention_layer_start 0 --block_attention_layer_end 40 --save_ckpt_log_name "$1" --pruner_type random --save_model
}
#task_noskip_origin(){
#  CUDA_VISIBLE_DEVICES=$2 python hf_prune.py --base_model "$4" --pruning_ratio 0.2 --device cpu --eval_device cuda --block_wise --block_mlp_layer_start 0 --block_mlp_layer_end 32 --block_attention_layer_start 0 --block_attention_layer_end 32 --save_ckpt_log_name "$1" --pruner_type taylor --taylor "$3" --save_model
#}
task_post_training(){
  CUDA_VISIBLE_DEVICES=$3 python post_training.py --base_model "$4" --prune_model prune_log/"$1"/pytorch_model.bin --data_path yahma/alpaca-cleaned --output_dir tune_log/"$2" --wandb_project "$2" --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64
}
#task_skip "$prune0" 1 'param_first' "$BASE_MODEL" & \
#task_skip_random_HD "$prune1" 2 "$BASE_MODEL" & \
task_noskip "$prune2" 2 'param_first' "$BASE_MODEL" & \
task_noskip_random_HD "$prune3" 3 "$BASE_MODEL"

wait
#CUDA_VISIBLE_DEVICES=1 bash $SCRIPT_B $BASE_MODEL prune_log/"$prune0" &
#CUDA_VISIBLE_DEVICES=2 bash $SCRIPT_B $BASE_MODEL prune_log/"$prune1" &
#CUDA_VISIBLE_DEVICES=3 bash $SCRIPT_B $BASE_MODEL prune_log/"$prune2" &
#CUDA_VISIBLE_DEVICES=0 bash $SCRIPT_B $BASE_MODEL prune_log/"$prune3"
wait

#task_post_training "$prune0" "$tune0" 1 "$BASE_MODEL" &
#task_post_training "$prune1" "$tune1" 2 "$BASE_MODEL" &
task_post_training "$prune2" "$tune2" 2 "$BASE_MODEL" &
task_post_training "$prune3" "$tune3" 3 "$BASE_MODEL"

wait

#CUDA_VISIBLE_DEVICES=1 bash $SCRIPT_A $BASE_MODEL tune_log/"$tune0" prune_log/"$prune0" 1400 &
#CUDA_VISIBLE_DEVICES=2 bash $SCRIPT_A $BASE_MODEL tune_log/"$tune1" prune_log/"$prune1" 1400 &
CUDA_VISIBLE_DEVICES=2 bash $SCRIPT_A $BASE_MODEL tune_log/"$tune2" prune_log/"$prune2" 1400 &
CUDA_VISIBLE_DEVICES=3 bash $SCRIPT_A $BASE_MODEL tune_log/"$tune3" prune_log/"$prune3" 1400

wait

#CUDA_VISIBLE_DEVICES=1 python generate.py --model_type tune_prune_LLM --ckpt prune_log/"$prune0"/pytorch_model.bin --lora_ckpt tune_log/"$tune0"/checkpoint-1400 --base_model "$BASE_MODEL" &
#CUDA_VISIBLE_DEVICES=2 python generate.py --model_type tune_prune_LLM --ckpt prune_log/"$prune1"/pytorch_model.bin --lora_ckpt tune_log/"$tune1"/checkpoint-1400 --base_model "$BASE_MODEL" &
CUDA_VISIBLE_DEVICES=2 python generate.py --model_type tune_prune_LLM --ckpt prune_log/"$prune2"/pytorch_model.bin --lora_ckpt tune_log/"$tune2"/checkpoint-1400 --base_model "$BASE_MODEL" &
CUDA_VISIBLE_DEVICES=3 python generate.py --model_type tune_prune_LLM --ckpt prune_log/"$prune3"/pytorch_model.bin --lora_ckpt tune_log/"$tune3"/checkpoint-1400 --base_model "$BASE_MODEL"
wait
echo "所有任务完成"