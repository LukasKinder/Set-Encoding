#!/bin/bash

# Models to run
models=(
  "mistralai/Mistral-7B-Instruct-v0.3:4"
  "mistralai/Mistral-7B-v0.3:4"
  "microsoft/Phi-3-mini-4k-instruct:4"
  "Aleph-Alpha/Pharia-1-LLM-7B-control-hf:4"
  "meta-llama/Meta-Llama-3-8B:2"
  "meta-llama/Llama-2-7b-hf:4"
  "tiiuae/falcon-7b:4"
  "gpt2:16"
  "meta-llama/Meta-Llama-3-8B-instruct:2"
  "meta-llama/Llama-2-7b-chat-hf:4"
  "tiiuae/falcon-7b-instruct:4"
)

# Orders to run 3
#orders=(
#  "entailment neutral contradiction"
#  "entailment contradiction neutral"
#  "contradiction entailment neutral"
#  "contradiction neutral entailment"
#  "neutral contradiction entailment"
#  "neutral entailment contradiction"
#)

# Orders to run 6
orders=(
  "entailment neutral contradiction entailment neutral contradiction"
  "contradiction neutral entailment contradiction neutral entailment"
  "entailment entailment neutral neutral contradiction contradiction"
  "contradiction contradiction entailment entailment neutral neutral"
  "neutral contradiction entailment entailment contradiction neutral"
  "neutral contradiction entailment contradiction neutral entailment"
)

# Run loop for each model and order with and without "--use_set_encoding"
for model_info in "${models[@]}"; do
  IFS=":" read -r model batch_size <<< "$model_info"
  
  for order in "${orders[@]}"; do
    sbatch run_generic.sh "$model" "$batch_size" "$order" "--use_set_encoding"
    sbatch run_generic.sh "$model" "$batch_size" "$order"
  done
done

exit 0
