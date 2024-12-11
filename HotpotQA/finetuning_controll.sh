# SAVE_PATH=$1
# MAX_TOKENS=$2
# NUM_EPOCHS=$3
# USE_SET_ENCODING=$4
# MODEL_ID=$5
# LEARNING_RATE=$6
# ACCUMLATION_STEPS=$7
# BATCH_SIZE=$8

sbatch generic_finetuning.sh finetuned_models/phi3-set 2000 3 True microsoft/Phi-3-mini-4k-instruct 1e-6 16 1
#sbatch generic_finetuning.sh finetuned_models/phi3 2000 3 False microsoft/Phi-3-mini-4k-instruct 1e-6 16 1

#sbatch generic_finetuning.sh finetuned_models/llama3-set 4000 3 True meta-llama/Meta-Llama-3-8B-instruct 3e-6 16 1
#sbatch generic_finetuning.sh finetuned_models/llama3 4000 3 False meta-llama/Meta-Llama-3-8B-instruct 3e-6 16 1