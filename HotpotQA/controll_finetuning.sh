#!/bin/bash
#SBATCH --time=0:00:10
#SBATCH --gres=gpu:0
#SBATCH --partition=L40S

# Arguments are: SAVE_PATH=$1 USE_SET_ENCODING=$2 MODEL_ID=$3 LEARNING_RATE=$4 ACCUMLATION_STEPS=$5  BATCH_SIZE=$5

sbatch generic_finetuning.sh ./finetuned_models/falcon-set True "tiiuae/falcon-7b-instruct" 3e-6 16 4 "0.25,0.25,0.25,0.25"
sbatch generic_finetuning.sh ./finetuned_models/falcon-normal False "tiiuae/falcon-7b-instruct" 3e-6 16 4 "0.25,0.25,0.25,0.25"

sbatch generic_finetuning.sh ./finetuned_models/falcon-normal-balance-1-0-0-0 False "tiiuae/falcon-7b-instruct" 3e-6 16 4 "1.0,0,0,0"
sbatch generic_finetuning.sh ./finetuned_models/falcon-normal-balance-7-1-1-1 False "tiiuae/falcon-7b-instruct" 3e-6 16 4 "0.7,0.1,0.1,0.1"
sbatch generic_finetuning.sh ./finetuned_models/falcon-normal-balance-3-2-3-2 False "tiiuae/falcon-7b-instruct" 3e-6 16 4 "0.3,0.2,0.3,0.2"

sbatch generic_finetuning.sh ./finetuned_models/falcon-set-balance-1-0-0-0 True "tiiuae/falcon-7b-instruct" 3e-6 16 4 "1.0,0,0,0"
sbatch generic_finetuning.sh ./finetuned_models/falcon-set-balance-7-1-1-1 True "tiiuae/falcon-7b-instruct" 3e-6 16 4 "0.7,0.1,0.1,0.1"
sbatch generic_finetuning.sh ./finetuned_models/falcon-set-balance-3-2-3-2 True "tiiuae/falcon-7b-instruct" 3e-6 16 4 "0.3,0.2,0.3,0.2"


#let's goooo:
#sbatch generic_finetuning.sh ./finetuned_models/llama-2-set True "meta-llama/Llama-2-7b-chat-hf" 5e-6 8 8
#sbatch generic_finetuning.sh ./finetuned_models/llama-2-normal False "meta-llama/Llama-2-7b-chat-hf" 5e-6 8 8
#sbatch generic_finetuning.sh ./finetuned_models/llama-3-set True "meta-llama/Meta-Llama-3-8B-instruct" 3e-6 8 8
#sbatch generic_finetuning.sh ./finetuned_models/llama-3-normal False "meta-llama/Meta-Llama-3-8B-instruct" 3e-6 8 8