#!/bin/bash

# Define models and balance settings
#declare -A models=(
#  ["falcon"]="tiiuae/falcon-7b-instruct"
#  ["mistral"]="mistralai/Mistral-7B-Instruct-v0.3"
#  ["llama2"]="meta-llama/Llama-2-7b-hf"
#  ["llama3"]="meta-llama/Meta-Llama-3-8B-instruct"
#  ["phi3"]="microsoft/Phi-3-mini-4k-instruct"
#)


# 4 2 2 2 2
#Falcon
sbatch generic_evaluation.sh "finetuned_models/falcon-normal-balance-4-2-2-2/checkpoint-3190" 16 99999 "option" "tiiuae/falcon-7b-instruct" "--first_tokens_differ"
sbatch generic_evaluation.sh "finetuned_models/falcon-set-balance-4-2-2-2/checkpoint-2895" 16 99999 "option" "tiiuae/falcon-7b-instruct" "--first_tokens_differ" "--use_set_encoding"
#Llama2
sbatch generic_evaluation.sh "finetuned_models/llama2-normal-balance-4-2-2-2/checkpoint-2785" 16 99999 "option" "meta-llama/Llama-2-7b-hf" "--first_tokens_differ"
sbatch generic_evaluation.sh "finetuned_models/llama2-set-balance-4-2-2-2/checkpoint-2535" 16 99999 "option" "meta-llama/Llama-2-7b-hf" "--first_tokens_differ" "--use_set_encoding"
#Llama3
sbatch generic_evaluation.sh "finetuned_models/llama3-normal-balance-4-2-2-2/checkpoint-3235" 8 99999 "option" "meta-llama/Meta-Llama-3-8B-instruct" "--first_tokens_differ"
sbatch generic_evaluation.sh "finetuned_models/llama3-set-balance-4-2-2-2/checkpoint-3015" 8 99999 "option" "meta-llama/Meta-Llama-3-8B-instruct" "--first_tokens_differ" "--use_set_encoding"
#Mistral
sbatch generic_evaluation.sh "finetuned_models/mistral-normal-balance-4-2-2-2/checkpoint-3025" 16 99999 "option" "mistralai/Mistral-7B-Instruct-v0.3" "--first_tokens_differ"
sbatch generic_evaluation.sh "finetuned_models/mistral-set-balance-4-2-2-2/checkpoint-2830" 16 99999 "option" "mistralai/Mistral-7B-Instruct-v0.3" "--first_tokens_differ" "--use_set_encoding"


exit 0

#Not trained
#Falcon
sbatch generic_evaluation.sh "tiiuae/falcon-7b-instruct" 16 99999 "option" "tiiuae/falcon-7b-instruct" "--fist_tokens_may_be_the_same"
sbatch generic_evaluation.sh "tiiuae/falcon-7b-instruct" 16 99999 "option" "tiiuae/falcon-7b-instruct" "--fist_tokens_may_be_the_same" "--use_set_encoding"
#Llama2
sbatch generic_evaluation.sh "meta-llama/Llama-2-7b-hf" 16 99999 "option" "meta-llama/Llama-2-7b-hf" "--fist_tokens_may_be_the_same"
sbatch generic_evaluation.sh "meta-llama/Llama-2-7b-hf" 16 99999 "option" "meta-llama/Llama-2-7b-hf" "--fist_tokens_may_be_the_same" "--use_set_encoding"
#Llama3
sbatch generic_evaluation.sh "meta-llama/Meta-Llama-3-8B-instruct" 16 99999 "option" "meta-llama/Meta-Llama-3-8B-instruct" "--fist_tokens_may_be_the_same"
sbatch generic_evaluation.sh "meta-llama/Meta-Llama-3-8B-instruct" 16 99999 "option" "meta-llama/Meta-Llama-3-8B-instruct" "--fist_tokens_may_be_the_same" "--use_set_encoding"
#Mistral
sbatch generic_evaluation.sh "mistralai/Mistral-7B-Instruct-v0.3" 16 99999 "option" "mistralai/Mistral-7B-Instruct-v0.3" "--fist_tokens_may_be_the_same"
sbatch generic_evaluation.sh "mistralai/Mistral-7B-Instruct-v0.3" 16 99999 "option" "mistralai/Mistral-7B-Instruct-v0.3" "--fist_tokens_may_be_the_same" "--use_set_encoding"

#0.25 0.25 0.25 0.25
#Falcon
sbatch generic_evaluation.sh "finetuned_models/savekeep-5-epochs/checkpoint-3180-falcon-normal-balanced" 16 99999 "option" "tiiuae/falcon-7b-instruct" "--fist_tokens_may_be_the_same"
sbatch generic_evaluation.sh "finetuned_models/savekeep-5-epochs/checkpoint-2890-falcon-set-balanced" 16 99999 "option" "tiiuae/falcon-7b-instruct" "--fist_tokens_may_be_the_same" "--use_set_encoding"
#Llama2
sbatch generic_evaluation.sh "finetuned_models/savekeep-5-epochs/checkpoint-2780-llama2-normal-balanced" 16 99999 "option" "meta-llama/Llama-2-7b-hf" "--fist_tokens_may_be_the_same"
sbatch generic_evaluation.sh "finetuned_models/savekeep-5-epochs/checkpoint-2530-llama2-set-balanced" 16 99999 "option" "meta-llama/Llama-2-7b-hf" "--fist_tokens_may_be_the_same" "--use_set_encoding"
#Llama3
sbatch generic_evaluation.sh "finetuned_models/savekeep-5-epochs/checkpoint-3230-llama3-normal-balanced" 16 99999 "option" "meta-llama/Meta-Llama-3-8B-instruct" "--fist_tokens_may_be_the_same"
sbatch generic_evaluation.sh "finetuned_models/savekeep-5-epochs/checkpoint-3010-llama3-set-balanced" 16 99999 "option" "meta-llama/Meta-Llama-3-8B-instruct" "--fist_tokens_may_be_the_same" "--use_set_encoding"
#Mistral
sbatch generic_evaluation.sh "finetuned_models/savekeep-5-epochs/checkpoint-3020-mistral-normal-balanced" 16 99999 "option" "mistralai/Mistral-7B-Instruct-v0.3" "--fist_tokens_may_be_the_same"
sbatch generic_evaluation.sh "finetuned_models/savekeep-5-epochs/checkpoint-2830-mistral-set-balanced" 16 99999 "option" "mistralai/Mistral-7B-Instruct-v0.3" "--fist_tokens_may_be_the_same" "--use_set_encoding"

# 3 2 3 2
#Falcon
sbatch generic_evaluation.sh "finetuned_models/falcon-normal-balance-3-2-3-2/checkpoint-3190" 16 99999 "option" "tiiuae/falcon-7b-instruct" "--fist_tokens_may_be_the_same"
sbatch generic_evaluation.sh "finetuned_models/falcon-set-balance-3-2-2-2/checkpoint-2895" 16 99999 "option" "tiiuae/falcon-7b-instruct" "--fist_tokens_may_be_the_same" "--use_set_encoding"
#Llama2
sbatch generic_evaluation.sh "finetuned_models/llama2-normal-balance-3-2-3-2/checkpoint-2785" 16 99999 "option" "meta-llama/Llama-2-7b-hf" "--fist_tokens_may_be_the_same"
sbatch generic_evaluation.sh "finetuned_models/llama2-set-balance-3-2-3-2/checkpoint-2535" 16 99999 "option" "meta-llama/Llama-2-7b-hf" "--fist_tokens_may_be_the_same" "--use_set_encoding"
#Llama3
sbatch generic_evaluation.sh "finetuned_models/llama3-normal-balance-3-2-3-2/checkpoint-3253" 16 99999 "option" "meta-llama/Meta-Llama-3-8B-instruct" "--fist_tokens_may_be_the_same"
sbatch generic_evaluation.sh "finetuned_models/llama3-set-balance-3-2-3-2/checkpoint-3015" 16 99999 "option" "meta-llama/Meta-Llama-3-8B-instruct" "--fist_tokens_may_be_the_same" "--use_set_encoding"
#Mistral
sbatch generic_evaluation.sh "finetuned_models/mistral-normal-balance-3-2-3-2/checkpoint-3025" 16 99999 "option" "mistralai/Mistral-7B-Instruct-v0.3" "--fist_tokens_may_be_the_same"
sbatch generic_evaluation.sh "finetuned_models/mistral-set-balance-3-2-3-2/checkpoint-2830" 16 99999 "option" "mistralai/Mistral-7B-Instruct-v0.3" "--fist_tokens_may_be_the_same" "--use_set_encoding"


# 7 1 1 1
#Falcon
sbatch generic_evaluation.sh "finetuned_models/falcon-normal-balance-7-1-1-1/checkpoint-3190" 16 99999 "option" "tiiuae/falcon-7b-instruct" "--fist_tokens_may_be_the_same"
sbatch generic_evaluation.sh "finetuned_models/falcon-set-balance-7-1-1-1/checkpoint-2895" 16 99999 "option" "tiiuae/falcon-7b-instruct" "--fist_tokens_may_be_the_same" "--use_set_encoding"
#Llama2
sbatch generic_evaluation.sh "finetuned_models/llama2-normal-balance-7-1-1-1/checkpoint-2785" 16 99999 "option" "meta-llama/Llama-2-7b-hf" "--fist_tokens_may_be_the_same"
sbatch generic_evaluation.sh "finetuned_models/llama2-set-balance-7-1-1-1/checkpoint-2535" 16 99999 "option" "meta-llama/Llama-2-7b-hf" "--fist_tokens_may_be_the_same" "--use_set_encoding"
#Llama3
sbatch generic_evaluation.sh "finetuned_models/llama3-normal-balance-7-1-1-1/checkpoint-3253" 16 99999 "option" "meta-llama/Meta-Llama-3-8B-instruct" "--fist_tokens_may_be_the_same"
sbatch generic_evaluation.sh "finetuned_models/llama3-set-balance-7-1-1-1/checkpoint-3015" 16 99999 "option" "meta-llama/Meta-Llama-3-8B-instruct" "--fist_tokens_may_be_the_same" "--use_set_encoding"
#Mistral
sbatch generic_evaluation.sh "finetuned_models/mistral-normal-balance-7-1-1-1/checkpoint-3025" 16 99999 "option" "mistralai/Mistral-7B-Instruct-v0.3" "--fist_tokens_may_be_the_same"
sbatch generic_evaluation.sh "finetuned_models/mistral-set-balance-7-1-1-1/checkpoint-2830" 16 99999 "option" "mistralai/Mistral-7B-Instruct-v0.3" "--fist_tokens_may_be_the_same" "--use_set_encoding"

# 1 0 0 0
#Falcon
sbatch generic_evaluation.sh "finetuned_models/falcon-normal-balance-1-0-0-0/checkpoint-3190" 16 99999 "option" "tiiuae/falcon-7b-instruct" "--fist_tokens_may_be_the_same"
sbatch generic_evaluation.sh "finetuned_models/falcon-set-balance-1-0-0-0/checkpoint-2895" 16 99999 "option" "tiiuae/falcon-7b-instruct" "--fist_tokens_may_be_the_same" "--use_set_encoding"
#Llama2
sbatch generic_evaluation.sh "finetuned_models/llama2-normal-balance-1-0-0-0/checkpoint-2785" 16 99999 "option" "meta-llama/Llama-2-7b-hf" "--fist_tokens_may_be_the_same"
sbatch generic_evaluation.sh "finetuned_models/llama2-set-balance-1-0-0-0/checkpoint-2535" 16 99999 "option" "meta-llama/Llama-2-7b-hf" "--fist_tokens_may_be_the_same" "--use_set_encoding"
#Llama3
sbatch generic_evaluation.sh "finetuned_models/llama3-normal-balance-1-0-0-0/checkpoint-3253" 16 99999 "option" "meta-llama/Meta-Llama-3-8B-instruct" "--fist_tokens_may_be_the_same"
sbatch generic_evaluation.sh "finetuned_models/llama3-set-balance-1-0-0-0/checkpoint-3015" 16 99999 "option" "meta-llama/Meta-Llama-3-8B-instruct" "--fist_tokens_may_be_the_same" "--use_set_encoding"
#Mistral
sbatch generic_evaluation.sh "finetuned_models/mistral-normal-balance-1-0-0-0/checkpoint-3025" 16 99999 "option" "mistralai/Mistral-7B-Instruct-v0.3" "--fist_tokens_may_be_the_same"
sbatch generic_evaluation.sh "finetuned_models/mistral-set-balance-1-0-0-0/checkpoint-2830" 16 99999 "option" "mistralai/Mistral-7B-Instruct-v0.3" "--fist_tokens_may_be_the_same" "--use_set_encoding"

exit 0

declare -A models=(
  ["falcon"]="tiiuae/falcon-7b-instruct"
  ["mistral"]="mistralai/Mistral-7B-Instruct-v0.3"
  ["llama2"]="meta-llama/Llama-2-7b-hf"
  ["llama3"]="meta-llama/Meta-Llama-3-8B-instruct"
)


#balances=("normal-balanced" "normal-balance-1-0-0-0" "normal-balance-3-2-3-2" "normal-balance-7-1-1-1" "set-balanced" "set-balance-1-0-0-0" "set-balance-3-2-3-2" "set-balance-7-1-1-1")
balances=("normal-balanced" "set-balanced")
#balances=("normal-balanced" "normal-balance-1-0-0-0" "normal-balance-3-2-3-2" "normal-balance-7-1-1-1" "set-balanced" "set-balance-1-0-0-0" "set-balance-3-2-3-2" "set-balance-7-1-1-1")

# Iterate over each model (falcon, mistral)
for model_key in "${!models[@]}"; do
  model_name="${models[$model_key]}"
  
  # Standard evaluation for the base model
  sbatch generic_evaluation.sh "$model_name" 8 99999 "option" "$model_name" "--fist_tokens_may_be_the_same" 
  sbatch generic_evaluation.sh "$model_name" 8 99999 "option" "$model_name" "--fist_tokens_may_be_the_same" "--use_set_encoding"
  
  # Iterate over each balance setting
  for balance in "${balances[@]}"; do
    # Check if the 'set' option should be included
    set_option=""/mnt/webscistorage/em7201/SetEncoding/MCQ/results/correct_evaluation
    [[ "$balance" == *"set"* ]] && set_option="--use_set_encoding"
    
    # Run evaluations for each checkpoint in the balance setting
    for checkpoint in finetuned_models/$model_key-$balance/checkpoint-*; do
      sbatch generic_evaluation.sh "$checkpoint" 8 99999 "option" "$model_name" "--fist_tokens_may_be_the_same" "$set_option"
    done
  done
done

exit 0

sbatch generic_evaluation.sh "tiiuae/falcon-7b-instruct" 16 99999 "option" "tiiuae/falcon-7b-instruct" "--first_tokens_differ"
sbatch generic_evaluation.sh "mistralai/Mistral-7B-Instruct-v0.3" 16 99999 "option" "mistralai/Mistral-7B-Instruct-v0.3" "--first_tokens_differ"

# "Usage: $0 <model> <batch_size> <max_batches> <eval_type> [<tokenizer_path>] [--first_tokens_differ] [--use_set_encoding] "

# falcon normal
sbatch sh_scripts/generic_evaluation.sh "tiiuae/falcon-7b-instruct" 32 99999 "option" "tiiuae/falcon-7b-instruct" "--first_tokens_differ"
for checkpoint in finetuned_models/falcon-normal-balanced/checkpoint-*; do
  sbatch sh_scripts/generic_evaluation.sh "$checkpoint" 32 99999 "option" "tiiuae/falcon-7b-instruct" "--first_tokens_differ"
done

# falcon normal balance 1-0-0-0
for checkpoint in finetuned_models/falcon-normal-balance-1-0-0-0/checkpoint-*; do
  sbatch sh_scripts/generic_evaluation.sh "$checkpoint" 32 99999 "option" "tiiuae/falcon-7b-instruct" "--first_tokens_differ"
done

# falcon normal balance 3-2-3-2
for checkpoint in finetuned_models/falcon-normal-balance-3-2-3-2/checkpoint-*; do
  sbatch sh_scripts/generic_evaluation.sh "$checkpoint" 32 99999 "option" "tiiuae/falcon-7b-instruct" "--first_tokens_differ"
done

# falcon normal balance 7-1-1-1
for checkpoint in finetuned_models/falcon-normal-balance-7-1-1-1/checkpoint-*; do
  sbatch sh_scripts/generic_evaluation.sh "$checkpoint" 32 99999 "option" "tiiuae/falcon-7b-instruct" "--first_tokens_differ"
done


# falcon set
sbatch sh_scripts/generic_evaluation.sh "tiiuae/falcon-7b-instruct" 32 99999 "option" "tiiuae/falcon-7b-instruct" "--first_tokens_differ" "--use_set_encoding"
for checkpoint in finetuned_models/falcon-set-balanced/checkpoint-*; do
  sbatch sh_scripts/generic_evaluation.sh "$checkpoint" 32 99999 "option" "tiiuae/falcon-7b-instruct" "--first_tokens_differ" "--use_set_encoding"
done

# falcon set balance 1-0-0-0
for checkpoint in finetuned_models/falcon-set-balance-1-0-0-0/checkpoint-*; do
  sbatch sh_scripts/generic_evaluation.sh "$checkpoint" 32 99999 "option" "tiiuae/falcon-7b-instruct" "--first_tokens_differ" "--use_set_encoding"
done

# falcon set balance 3-2-3-2
for checkpoint in finetuned_models/falcon-set-balance-3-2-3-2/checkpoint-*; do
  sbatch sh_scripts/generic_evaluation.sh "$checkpoint" 32 99999 "option" "tiiuae/falcon-7b-instruct" "--first_tokens_differ" "--use_set_encoding"
done

# falcon set balance 7-1-1-1
for checkpoint in finetuned_models/falcon-set-balance-7-1-1-1/checkpoint-*; do
  sbatch sh_scripts/generic_evaluation.sh "$checkpoint" 32 99999 "option" "tiiuae/falcon-7b-instruct" "--first_tokens_differ" "--use_set_encoding"
done


# mistral normal
sbatch sh_scripts/generic_evaluation.sh "mistralai/Mistral-7B-Instruct-v0.3" 32 99999 "option" "mistralai/Mistral-7B-Instruct-v0.3" "--first_tokens_differ"
for checkpoint in finetuned_models/mistral-normal-balanced/checkpoint-*; do
  sbatch sh_scripts/generic_evaluation.sh "$checkpoint" 32 99999 "option" "mistralai/Mistral-7B-Instruct-v0.3" "--first_tokens_differ"
done

# mistral normal balance 1-0-0-0
for checkpoint in finetuned_models/mistral-normal-balance-1-0-0-0/checkpoint-*; do
  sbatch sh_scripts/generic_evaluation.sh "$checkpoint" 32 99999 "option" "mistralai/Mistral-7B-Instruct-v0.3" "--first_tokens_differ"
done

# mistral normal balance 3-2-3-2
for checkpoint in finetuned_models/mistral-normal-balance-3-2-3-2/checkpoint-*; do
  sbatch sh_scripts/generic_evaluation.sh "$checkpoint" 32 99999 "option" "mistralai/Mistral-7B-Instruct-v0.3" "--first_tokens_differ"
done

# mistral normal balance 7-1-1-1
for checkpoint in finetuned_models/mistral-normal-balance-7-1-1-1/checkpoint-*; do
  sbatch sh_scripts/generic_evaluation.sh "$checkpoint" 32 99999 "option" "mistralai/Mistral-7B-Instruct-v0.3" "--first_tokens_differ"
done


# mistral set
sbatch sh_scripts/generic_evaluation.sh "mistralai/Mistral-7B-Instruct-v0.3" 32 99999 "option" "mistralai/Mistral-7B-Instruct-v0.3" "--first_tokens_differ" "--use_set_encoding"
for checkpoint in finetuned_models/mistral-set/checkpoint-*; do
  sbatch sh_scripts/generic_evaluation.sh "$checkpoint" 32 99999 "option" "mistralai/Mistral-7B-Instruct-v0.3" "--first_tokens_differ" "--use_set_encoding"
done

# mistral set balance 1-0-0-0
for checkpoint in finetuned_models/mistral-set-balance-1-0-0-0/checkpoint-*; do
  sbatch sh_scripts/generic_evaluation.sh "$checkpoint" 32 99999 "option" "mistralai/Mistral-7B-Instruct-v0.3" "--first_tokens_differ" "--use_set_encoding"
done

# mistral set balance 3-2-3-2
for checkpoint in finetuned_models/mistral-set-balance-3-2-3-2/checkpoint-*; do
  sbatch sh_scripts/generic_evaluation.sh "$checkpoint" 32 99999 "option" "mistralai/Mistral-7B-Instruct-v0.3" "--first_tokens_differ" "--use_set_encoding"
done

# mistral set balance 7-1-1-1
for checkpoint in finetuned_models/mistral-set-balance-7-1-1-1/checkpoint-*; do
  sbatch sh_scripts/generic_evaluation.sh "$checkpoint" 32 99999 "option" "mistralai/Mistral-7B-Instruct-v0.3" "--first_tokens_differ" "--use_set_encoding"
done
