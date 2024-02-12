#
#prune_ckpt_path='llama_prune_0.2_without_tune_block_param_second'
#tune_ckpt_path='llama_prune_0.2_with_tune_block_param_second'
#echo "[START] - Start Pruning Model"
#CUDA_VISIBLE_DEVICES=1 python hf_prune.py --pruning_ratio 0.25 --device cpu  --eval_device cuda --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 30 --block_attention_layer_start 4 --block_attention_layer_end 30 --save_ckpt_log_name $prune_ckpt_path --pruner_type taylor --test_after_train --taylor param_second --save_model
#echo "[FINISH] - Finish Pruning Model"
#echo "[START] - Start Tuning"
#CUDA_VISIBLE_DEVICES=1 python post_training.py --prune_model prune_log/$prune_ckpt_path/pytorch_model.bin --data_path yahma/alpaca-cleaned --output_dir tune_log/$tune_ckpt_path --wandb_project $tune_ckpt_path --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64
#echo "[FINISH] - Finish Prune and Post-Training."

prune_ckpt_path='llama_prune_0.2_without_tune_channel_param_mix'
tune_ckpt_path='llama_prune_0.2_with_tune_channel_param_mix'
echo "[START] - Start Pruning Model"
CUDA_VISIBLE_DEVICES=2 python hf_prune.py --pruning_ratio 0.2 --device cpu  --eval_device cuda --channel_wise --save_ckpt_log_name $prune_ckpt_path --pruner_type taylor --test_after_train --taylor param_mix --save_model
echo "[FINISH] - Finish Pruning Model"
echo "[START] - Start Tuning"
CUDA_VISIBLE_DEVICES=2 python post_training.py --prune_model prune_log/$prune_ckpt_path/pytorch_model.bin --data_path yahma/alpaca-cleaned --output_dir tune_log/$tune_ckpt_path --wandb_project $tune_ckpt_path --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64
echo "[FINISH] - Finish Prune and Post-Training."

prune_ckpt_path='llama_prune_0.2_without_tune_channel_param_second'
tune_ckpt_path='llama_prune_0.2_with_tune_channel_param_second'
echo "[START] - Start Pruning Model"
CUDA_VISIBLE_DEVICES=1 python hf_prune.py --pruning_ratio 0.2 --device cpu  --eval_device cuda --channel_wise --save_ckpt_log_name $prune_ckpt_path --pruner_type taylor --test_after_train --taylor param_second --save_model
echo "[FINISH] - Finish Pruning Model"
echo "[START] - Start Tuning"
CUDA_VISIBLE_DEVICES=1 python post_training.py --prune_model prune_log/$prune_ckpt_path/pytorch_model.bin --data_path yahma/alpaca-cleaned --output_dir tune_log/$tune_ckpt_path --wandb_project $tune_ckpt_path --lora_r 8 --num_epochs 2 --learning_rate 1e-4 --batch_size 64
echo "[FINISH] - Finish Prune and Post-Training."