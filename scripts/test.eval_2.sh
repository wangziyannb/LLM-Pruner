SCRIPT_A='scripts/a.sh'
BASE_MODEL='baffo32/decapoda-research-llama-7B-hf'
CUDA_VISIBLE_DEVICES=2 bash $SCRIPT_A $BASE_MODEL tune_log/llama_prune_0.2_with_tune_Vector_taylor prune_log/llama_prune_0.2_without_tune_Vector_taylor 1400
CUDA_VISIBLE_DEVICES=2 bash $SCRIPT_A $BASE_MODEL tune_log/llama_prune_0.2_with_tune_Channel_taylor prune_log/llama_prune_0.2_without_tune_Channel_taylor 1400

