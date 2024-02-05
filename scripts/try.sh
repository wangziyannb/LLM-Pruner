#CUDA_VISIBLE_DEVICES=3 python generate.py --model_type tune_prune_LLM --ckpt prune_log/llama_prune_0.2_without_tune_Channel_taylor/pytorch_model.bin --lora_ckpt tune_log/llama_prune_0.2_with_tune_Channel_taylor/checkpoint-1400 --base_model baffo32/decapoda-research-llama-7B-hf
prune_ckpt_path='llama_prune_0.2_without_tune_test'
tune_ckpt_path='llama_prune_0.2_with_tune_test'
echo "[START] - Start Pruning Model"
CUDA_VISIBLE_DEVICES=3 python hf_prune.py --pruning_ratio 0.25 --device cuda  --eval_device cuda --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 30 --block_attention_layer_start 4 --block_attention_layer_end 30 --save_ckpt_log_name $prune_ckpt_path --pruner_type taylor --test_after_train --taylor param_mix --save_model
#echo "[FINISH] - Finish Pruning Model"
#echo "[START] - Start Tuning"
#CUDA_VISIBLE_DEVICES=3 python post_training.py --prune_model prune_log/$prune_ckpt_path/pytorch_model.bin --data_path yahma/alpaca-cleaned --output_dir tune_log/$tune_ckpt_path --wandb_project $tune_ckpt_path --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64
#echo "[FINISH] - Finish Prune and Post-Training."