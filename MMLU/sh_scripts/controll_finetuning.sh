
# Arguments are: SAVE_PATH=$1 USE_SET_ENCODING=$2 MODEL_ID=$3 LEARNING_RATE=$4 ACCUMLATION_STEPS=$5  BATCH_SIZE=$5

#leftover
sbatch generic_finetuning.sh ./finetuned_models/llama3-set-balance-4-2-2-2 True "meta-llama/Meta-Llama-3-8B-instruct" 2e-6 8 4 "0.4,0.2,0.2,0.2"
sbatch generic_finetuning.sh ./finetuned_models/llama2-set-balance-4-2-2-2 True "meta-llama/Llama-2-7b-hf" 2e-6 8 4 "0.4,0.2,0.2,0.2"

exit 0

#new 4,2,2,2 condition
sbatch generic_finetuning.sh ./finetuned_models/llama2-set-balance-4-2-2-2 True "meta-llama/Llama-2-7b-hf" 2e-6 8 4 "0.4,0.2,0.2,0.2"
sbatch generic_finetuning.sh ./finetuned_models/llama2-normal-balance-4-2-2-2 False "meta-llama/Llama-2-7b-hf" 2e-6 8 4 "0.4,0.2,0.2,0.2"

sbatch generic_finetuning.sh ./finetuned_models/llama3-normal-balance-4-2-2-2 False "meta-llama/Meta-Llama-3-8B-instruct" 2e-6 8 4 "0.4,0.2,0.2,0.2"
sbatch generic_finetuning.sh ./finetuned_models/llama3-set-balance-4-2-2-2 True "meta-llama/Meta-Llama-3-8B-instruct" 2e-6 8 4 "0.4,0.2,0.2,0.2"

sbatch generic_finetuning.sh ./finetuned_models/mistral-set-balance-4-2-2-2 True "mistralai/Mistral-7B-Instruct-v0.3" 2e-6 8 4 "0.4,0.2,0.2,0.2"
sbatch generic_finetuning.sh ./finetuned_models/mistral-normal-balance-4-2-2-2 False "mistralai/Mistral-7B-Instruct-v0.3" 2e-6 8 4 "0.4,0.2,0.2,0.2"

sbatch generic_finetuning.sh ./finetuned_models/falcon-set-balance-4-2-2-2 True "tiiuae/falcon-7b-instruct" 2e-6 8 4 "0.4,0.2,0.2,0.2"
sbatch generic_finetuning.sh ./finetuned_models/falcon-normal-balance-4-2-2-2 False "tiiuae/falcon-7b-instruct" 2e-6 8 4 "0.4,0.2,0.2,0.2"


exit 0


# Falcon set
sbatch generic_finetuning.sh ./finetuned_models/falcon-set-balance-1-0-0-0 True "tiiuae/falcon-7b-instruct" 2e-6 8 4 "1.0,0,0,0"
sbatch generic_finetuning.sh ./finetuned_models/falcon-set-balance-7-1-1-1 True "tiiuae/falcon-7b-instruct" 2e-6 8 4 "0.7,0.1,0.1,0.1"
sbatch generic_finetuning.sh ./finetuned_models/falcon-set-balance-3-2-3-2 True "tiiuae/falcon-7b-instruct" 2e-6 8 4 "0.3,0.2,0.3,0.2"
#sbatch generic_finetuning.sh ./finetuned_models/falcon-set-balanced True "tiiuae/falcon-7b-instruct" 2e-6 8 4 "0.25,0.25,0.25,0.25"

# Falcon normal
sbatch generic_finetuning.sh ./finetuned_models/falcon-normal-balance-1-0-0-0 False "tiiuae/falcon-7b-instruct" 2e-6 8 4 "1.0,0,0,0"
sbatch generic_finetuning.sh ./finetuned_models/falcon-normal-balance-7-1-1-1 False "tiiuae/falcon-7b-instruct" 2e-6 8 4 "0.7,0.1,0.1,0.1"
sbatch generic_finetuning.sh ./finetuned_models/falcon-normal-balance-3-2-3-2 False "tiiuae/falcon-7b-instruct" 2e-6 8 4 "0.3,0.2,0.3,0.2"
#sbatch generic_finetuning.sh ./finetuned_models/falcon-normal-balanced False "tiiuae/falcon-7b-instruct" 2e-6 8 4 "0.25,0.25,0.25,0.25"


# Llama2 set
sbatch generic_finetuning.sh ./finetuned_models/llama2-set-balance-1-0-0-0 True "meta-llama/Llama-2-7b-hf" 2e-6 8 4 "1.0,0,0,0"
sbatch generic_finetuning.sh ./finetuned_models/llama2-set-balance-7-1-1-1 True "meta-llama/Llama-2-7b-hf" 2e-6 8 4 "0.7,0.1,0.1,0.1"
sbatch generic_finetuning.sh ./finetuned_models/llama2-set-balance-3-2-3-2 True "meta-llama/Llama-2-7b-hf" 2e-6 8 4 "0.3,0.2,0.3,0.2"
#sbatch generic_finetuning.sh ./finetuned_models/llama2-set-balanced True "meta-llama/Llama-2-7b-hf" 2e-6 8 4 "0.25,0.25,0.25,0.25"

# Llama2 normal
sbatch generic_finetuning.sh ./finetuned_models/llama2-normal-balance-1-0-0-0 False "meta-llama/Llama-2-7b-hf" 2e-6 8 4 "1.0,0,0,0"
sbatch generic_finetuning.sh ./finetuned_models/llama2-normal-balance-7-1-1-1 False "meta-llama/Llama-2-7b-hf" 2e-6 8 4 "0.7,0.1,0.1,0.1"
sbatch generic_finetuning.sh ./finetuned_models/llama2-normal-balance-3-2-3-2 False "meta-llama/Llama-2-7b-hf" 2e-6 8 4 "0.3,0.2,0.3,0.2"
#sbatch generic_finetuning.sh ./finetuned_models/llama2-normal-balanced False "meta-llama/Llama-2-7b-hf" 2e-6 8 4 "0.25,0.25,0.25,0.25"

# Llama3 set
sbatch generic_finetuning.sh ./finetuned_models/llama3-set-balance-1-0-0-0 True "meta-llama/Meta-Llama-3-8B-instruct" 2e-6 8 4 "1.0,0,0,0"
sbatch generic_finetuning.sh ./finetuned_models/llama3-set-balance-7-1-1-1 True "meta-llama/Meta-Llama-3-8B-instruct" 2e-6 8 4 "0.7,0.1,0.1,0.1"
sbatch generic_finetuning.sh ./finetuned_models/llama3-set-balance-3-2-3-2 True "meta-llama/Meta-Llama-3-8B-instruct" 2e-6 8 4 "0.3,0.2,0.3,0.2"
#sbatch generic_finetuning.sh ./finetuned_models/llama3-set-balanced True "meta-llama/Meta-Llama-3-8B-instruct" 2e-6 8 4 "0.25,0.25,0.25,0.25"

# Llama3 normal
sbatch generic_finetuning.sh ./finetuned_models/llama3-normal-balance-1-0-0-0 False "meta-llama/Meta-Llama-3-8B-instruct" 2e-6 8 4 "1.0,0,0,0"
sbatch generic_finetuning.sh ./finetuned_models/llama3-normal-balance-7-1-1-1 False "meta-llama/Meta-Llama-3-8B-instruct" 2e-6 8 4 "0.7,0.1,0.1,0.1"
sbatch generic_finetuning.sh ./finetuned_models/llama3-normal-balance-3-2-3-2 False "meta-llama/Meta-Llama-3-8B-instruct" 2e-6 8 4 "0.3,0.2,0.3,0.2"
#sbatch generic_finetuning.sh ./finetuned_models/llama3-normal-balanced False "meta-llama/Meta-Llama-3-8B-instruct" 2e-6 8 4 "0.25,0.25,0.25,0.25"


# Mistral set
sbatch generic_finetuning.sh ./finetuned_models/mistral-set-balance-1-0-0-0 True "mistralai/Mistral-7B-Instruct-v0.3" 2e-6 8 4 "1.0,0,0,0"
sbatch generic_finetuning.sh ./finetuned_models/mistral-set-balance-7-1-1-1 True "mistralai/Mistral-7B-Instruct-v0.3" 2e-6 8 4 "0.7,0.1,0.1,0.1"
sbatch generic_finetuning.sh ./finetuned_models/mistral-set-balance-3-2-3-2 True "mistralai/Mistral-7B-Instruct-v0.3" 2e-6 8 4 "0.3,0.2,0.3,0.2"
#sbatch generic_finetuning.sh ./finetuned_models/mistral-set-balanced True "mistralai/Mistral-7B-Instruct-v0.3" 2e-6 8 4 "0.25,0.25,0.25,0.25"

# Mistral normal
sbatch generic_finetuning.sh ./finetuned_models/mistral-normal-balance-1-0-0-0 False "mistralai/Mistral-7B-Instruct-v0.3" 2e-6 8 4 "1.0,0,0,0"
sbatch generic_finetuning.sh ./finetuned_models/mistral-normal-balance-7-1-1-1 False "mistralai/Mistral-7B-Instruct-v0.3" 2e-6 8 4 "0.7,0.1,0.1,0.1"
sbatch generic_finetuning.sh ./finetuned_models/mistral-normal-balance-3-2-3-2 False "mistralai/Mistral-7B-Instruct-v0.3" 2e-6 8 4 "0.3,0.2,0.3,0.2"
#sbatch generic_finetuning.sh ./finetuned_models/mistral-normal-balanced False "mistralai/Mistral-7B-Instruct-v0.3" 2e-6 8 4 "0.25,0.25,0.25,0.25"


# Phi3 set
sbatch generic_finetuning.sh ./finetuned_models/phi3-set-balance-1-0-0-0 True "microsoft/Phi-3-mini-4k-instruct" 1e-6 8 4 "1.0,0,0,0"
sbatch generic_finetuning.sh ./finetuned_models/phi3-set-balance-7-1-1-1 True "microsoft/Phi-3-mini-4k-instruct" 1e-6 8 4 "0.7,0.1,0.1,0.1"
sbatch generic_finetuning.sh ./finetuned_models/phi3-set-balance-3-2-3-2 True "microsoft/Phi-3-mini-4k-instruct" 1e-6 8 4 "0.3,0.2,0.3,0.2"
sbatch generic_finetuning.sh ./finetuned_models/phi3-set-balanced True "microsoft/Phi-3-mini-4k-instruct" 1e-6 8 4 "0.25,0.25,0.25,0.25"

# Phi3 normal
sbatch generic_finetuning.sh ./finetuned_models/phi3-normal-balance-1-0-0-0 False "microsoft/Phi-3-mini-4k-instruct" 1e-6 8 4 "1.0,0,0,0"
sbatch generic_finetuning.sh ./finetuned_models/phi3-normal-balance-7-1-1-1 False "microsoft/Phi-3-mini-4k-instruct" 1e-6 8 4 "0.7,0.1,0.1,0.1"
sbatch generic_finetuning.sh ./finetuned_models/phi3-normal-balance-3-2-3-2 False "microsoft/Phi-3-mini-4k-instruct" 1e-6 8 4 "0.3,0.2,0.3,0.2"
sbatch generic_finetuning.sh ./finetuned_models/phi3-normal-balanced False "microsoft/Phi-3-mini-4k-instruct" 1e-6 8 4 "0.25,0.25,0.25,0.25"
