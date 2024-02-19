SCRIPT_B='scripts/b.sh'
BASE_MODEL='baffo32/decapoda-research-llama-7B-hf'

CUDA_VISIBLE_DEVICES=3 bash $SCRIPT_B $BASE_MODEL prune_log/llama_prune_0.2_without_tune_block_L2
CUDA_VISIBLE_DEVICES=3 bash $SCRIPT_B $BASE_MODEL prune_log/llama_prune_0.2_without_tune_block_Random

