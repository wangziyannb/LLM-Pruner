SCRIPT_A='scripts/a.sh'
BASE_MODEL='baffo32/decapoda-research-llama-7B-hf'

CUDA_VISIBLE_DEVICES=1 bash $SCRIPT_A $BASE_MODEL tune_log/llama_prune_0.2_with_tune_channel_vectorize prune_log/llama_prune_0.2_without_tune_channel_vectorize 1400
CUDA_VISIBLE_DEVICES=1 bash $SCRIPT_A $BASE_MODEL tune_log/llama_prune_0.2_with_tune_block_vectorize prune_log/llama_prune_0.2_without_tune_block_vectorize 1400
