#!/bin/bash

#MODEL=$1
#BATCH_SIZE=$2
#MAX_TOKENS=$3
#TOKENIZER_PATH=$4
#USE_SET_ENCODING=$4 

sbatch run_generic.sh meta-llama/Meta-Llama-3-8B 1 17000 meta-llama/Meta-Llama-3-8B-instruct --use_set_encoding
sbatch run_generic.sh meta-llama/Meta-Llama-3-8B 1 17000 meta-llama/Meta-Llama-3-8B-instruct

sbatch run_generic.sh finetuned_models/llama3-set/checkpoint-2400 1 17000 meta-llama/Meta-Llama-3-8B-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/llama3/checkpoint-2280 1 17000 meta-llama/Meta-Llama-3-8B-instruct

exit 0

sbatch run_generic.sh finetuned_models/phi3-set/checkpoint-240 1 6500 microsoft/Phi-3-mini-4k-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/phi3-set/checkpoint-480 1 6500 microsoft/Phi-3-mini-4k-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/phi3-set/checkpoint-720 1 6500 microsoft/Phi-3-mini-4k-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/phi3-set/checkpoint-960 1 6500 microsoft/Phi-3-mini-4k-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/phi3-set/checkpoint-1200 1 6500 microsoft/Phi-3-mini-4k-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/phi3-set/checkpoint-1440 1 6500 microsoft/Phi-3-mini-4k-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/phi3-set/checkpoint-1680 1 6500 microsoft/Phi-3-mini-4k-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/phi3-set/checkpoint-1920 1 6500 microsoft/Phi-3-mini-4k-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/phi3-set/checkpoint-2160 1 6500 microsoft/Phi-3-mini-4k-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/phi3-set/checkpoint-2400 1 6500 microsoft/Phi-3-mini-4k-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/phi3-set/checkpoint-2640 1 6500 microsoft/Phi-3-mini-4k-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/phi3-set/checkpoint-2880 1 6500 microsoft/Phi-3-mini-4k-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/phi3-set/checkpoint-3120 1 6500 microsoft/Phi-3-mini-4k-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/phi3-set/checkpoint-3360 1 6500 microsoft/Phi-3-mini-4k-instruct --use_set_encoding

exit 0


sbatch run_generic.sh finetuned_models/phi3/checkpoint-240 1 6500 microsoft/Phi-3-mini-4k-instruct
sbatch run_generic.sh finetuned_models/phi3-set/checkpoint-240 1 6500 microsoft/Phi-3-mini-4k-instruct --use_set_encoding

sbatch run_generic.sh finetuned_models/llama3-set/checkpoint-120 1 17000 meta-llama/Meta-Llama-3-8B-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/llama3-set/checkpoint-240 1 17000 meta-llama/Meta-Llama-3-8B-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/llama3-set/checkpoint-360 1 17000 meta-llama/Meta-Llama-3-8B-instruct --use_set_encoding
#sbatch run_generic.sh finetuned_models/llama3-set/checkpoint-480 1 17000 meta-llama/Meta-Llama-3-8B-instruct
sbatch run_generic.sh finetuned_models/llama3-set/checkpoint-600 1 17000 meta-llama/Meta-Llama-3-8B-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/llama3-set/checkpoint-720 1 17000 meta-llama/Meta-Llama-3-8B-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/llama3-set/checkpoint-840 1 17000 meta-llama/Meta-Llama-3-8B-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/llama3-set/checkpoint-960 1 17000 meta-llama/Meta-Llama-3-8B-instruct --use_set_encoding
#sbatch run_generic.sh finetuned_models/llama3/checkpoint-1080 1 17000 meta-llama/Meta-Llama-3-8B-instruct
sbatch run_generic.sh finetuned_models/llama3-set/checkpoint-1200 1 17000 meta-llama/Meta-Llama-3-8B-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/llama3-set/checkpoint-1320 1 17000 meta-llama/Meta-Llama-3-8B-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/llama3-set/checkpoint-1440 1 17000 meta-llama/Meta-Llama-3-8B-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/llama3-set/checkpoint-1560 1 17000 meta-llama/Meta-Llama-3-8B-instruct --use_set_encoding
#sbatch run_generic.sh finetuned_models/llama3/checkpoint-1680 1 17000 meta-llama/Meta-Llama-3-8B-instruct
sbatch run_generic.sh finetuned_models/llama3-set/checkpoint-1800 1 17000 meta-llama/Meta-Llama-3-8B-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/llama3-set/checkpoint-1920 1 17000 meta-llama/Meta-Llama-3-8B-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/llama3-set/checkpoint-2040 1 17000 meta-llama/Meta-Llama-3-8B-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/llama3-set/checkpoint-2160 1 17000 meta-llama/Meta-Llama-3-8B-instruct --use_set_encoding
#sbatch run_generic.sh finetuned_models/llama3/checkpoint-2280 1 17000 meta-llama/Meta-Llama-3-8B-instruct

sbatch run_generic.sh finetuned_models/phi3/checkpoint-480 1 6500 microsoft/Phi-3-mini-4k-instruct
sbatch run_generic.sh finetuned_models/phi3/checkpoint-720 1 6500 microsoft/Phi-3-mini-4k-instruct
sbatch run_generic.sh finetuned_models/phi3/checkpoint-960 1 6500 microsoft/Phi-3-mini-4k-instruct
sbatch run_generic.sh finetuned_models/phi3/checkpoint-1200 1 6500 microsoft/Phi-3-mini-4k-instruct
sbatch run_generic.sh finetuned_models/phi3/checkpoint-1440 1 6500 microsoft/Phi-3-mini-4k-instruct
sbatch run_generic.sh finetuned_models/phi3/checkpoint-1680 1 6500 microsoft/Phi-3-mini-4k-instruct
sbatch run_generic.sh finetuned_models/phi3/checkpoint-1920 1 6500 microsoft/Phi-3-mini-4k-instruct
sbatch run_generic.sh finetuned_models/phi3/checkpoint-2160 1 6500 microsoft/Phi-3-mini-4k-instruct
sbatch run_generic.sh finetuned_models/phi3/checkpoint-2400 1 6500 microsoft/Phi-3-mini-4k-instruct
sbatch run_generic.sh finetuned_models/phi3/checkpoint-2640 1 6500 microsoft/Phi-3-mini-4k-instruct
sbatch run_generic.sh finetuned_models/phi3/checkpoint-2880 1 6500 microsoft/Phi-3-mini-4k-instruct
sbatch run_generic.sh finetuned_models/phi3/checkpoint-3120 1 6500 microsoft/Phi-3-mini-4k-instruct
sbatch run_generic.sh finetuned_models/phi3/checkpoint-3360 1 6500 microsoft/Phi-3-mini-4k-instruct
sbatch run_generic.sh finetuned_models/phi3/checkpoint-3600 1 6500 microsoft/Phi-3-mini-4k-instruct
sbatch run_generic.sh finetuned_models/phi3/checkpoint-3840 1 6500 microsoft/Phi-3-mini-4k-instruct
sbatch run_generic.sh finetuned_models/phi3/checkpoint-4080 1 6500 microsoft/Phi-3-mini-4k-instruct
sbatch run_generic.sh finetuned_models/phi3/checkpoint-4320 1 6500 microsoft/Phi-3-mini-4k-instruct
sbatch run_generic.sh finetuned_models/phi3/checkpoint-4560 1 6500 microsoft/Phi-3-mini-4k-instruct
sbatch run_generic.sh finetuned_models/phi3/checkpoint-4800 1 6500 microsoft/Phi-3-mini-4k-instruct
sbatch run_generic.sh finetuned_models/phi3/checkpoint-5040 1 6500 microsoft/Phi-3-mini-4k-instruct
sbatch run_generic.sh finetuned_models/phi3/checkpoint-5280 1 6500 microsoft/Phi-3-mini-4k-instruct
sbatch run_generic.sh finetuned_models/phi3/checkpoint-5520 1 6500 microsoft/Phi-3-mini-4k-instruct

sbatch run_generic.sh finetuned_models/phi3-set/checkpoint-240 1 6500 microsoft/Phi-3-mini-4k-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/phi3-set/checkpoint-480 1 6500 microsoft/Phi-3-mini-4k-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/phi3-set/checkpoint-720 1 6500 microsoft/Phi-3-mini-4k-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/phi3-set/checkpoint-960 1 6500 microsoft/Phi-3-mini-4k-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/phi3-set/checkpoint-1200 1 6500 microsoft/Phi-3-mini-4k-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/phi3-set/checkpoint-1440 1 6500 microsoft/Phi-3-mini-4k-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/phi3-set/checkpoint-1680 1 6500 microsoft/Phi-3-mini-4k-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/phi3-set/checkpoint-1920 1 6500 microsoft/Phi-3-mini-4k-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/phi3-set/checkpoint-2160 1 6500 microsoft/Phi-3-mini-4k-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/phi3-set/checkpoint-2400 1 6500 microsoft/Phi-3-mini-4k-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/phi3-set/checkpoint-2640 1 6500 microsoft/Phi-3-mini-4k-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/phi3-set/checkpoint-2880 1 6500 microsoft/Phi-3-mini-4k-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/phi3-set/checkpoint-3120 1 6500 microsoft/Phi-3-mini-4k-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/phi3-set/checkpoint-3360 1 6500 microsoft/Phi-3-mini-4k-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/phi3-set/checkpoint-3600 1 6500 microsoft/Phi-3-mini-4k-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/phi3-set/checkpoint-3840 1 6500 microsoft/Phi-3-mini-4k-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/phi3-set/checkpoint-4080 1 6500 microsoft/Phi-3-mini-4k-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/phi3-set/checkpoint-4320 1 6500 microsoft/Phi-3-mini-4k-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/phi3-set/checkpoint-4560 1 6500 microsoft/Phi-3-mini-4k-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/phi3-set/checkpoint-4800 1 6500 microsoft/Phi-3-mini-4k-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/phi3-set/checkpoint-5040 1 6500 microsoft/Phi-3-mini-4k-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/phi3-set/checkpoint-5280 1 6500 microsoft/Phi-3-mini-4k-instruct --use_set_encoding
sbatch run_generic.sh finetuned_models/phi3-set/checkpoint-5520 1 6500 microsoft/Phi-3-mini-4k-instruct --use_set_encoding


exit 0

# Models to run
models=(
  "tiiuae/falcon-7b:6500"
  "meta-llama/Meta-Llama-3-8B-instruct:16000"
  "microsoft/Phi-3-mini-4k-instruct:6500"
  "Aleph-Alpha/Pharia-1-LLM-7B-control-hf:6000"
  "meta-llama/Llama-2-7b-hf:16000"
)

# Run loop for each model and order with and without "--use_set_encoding"
for model_info in "${models[@]}"; do
  IFS=":" read -r model max_tokens <<< "$model_info"
  
  sbatch run_generic.sh "$model" "1" "$max_tokens"  "--use_set_encoding"
  sbatch run_generic.sh "$model" "1" "$max_tokens" 

done

# Models to run
models=(
  "mistralai/Mistral-7B-Instruct-v0.3:1"
  "mistralai/Mistral-7B-v0.3:1"
  "microsoft/Phi-3-mini-4k-instruct:1"
  "Aleph-Alpha/Pharia-1-LLM-7B-control-hf:1"
  "meta-llama/Meta-Llama-3-8B:1"
  "meta-llama/Llama-2-7b-hf:1"
  "tiiuae/falcon-7b:1"
  "gpt2:2"
  "meta-llama/Meta-Llama-3-8B-instruct:1"
  "meta-llama/Llama-2-7b-chat-hf:2"
  "tiiuae/falcon-7b-instruct:1"
)

