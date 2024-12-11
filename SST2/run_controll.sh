
# "meta-llama/Llama-2-7b-chat-hf", "meta-llama/Meta-Llama-3-8B-instruct","mistralai/Mistral-7B-Instruct-v0.3","tiiuae/falcon-7b-instruct"

# "meta-llama/Llama-2-7b-hf", "meta-llama/Meta-Llama-3-8B", "gpt2","mistralai/Mistral-7B-v0.3","tiiuae/falcon-7b"

# phi 3
sbatch run_generic.sh Aleph-Alpha/Pharia-1-LLM-7B-control-hf 8 "Positive Positive Negative Negative"
sbatch run_generic.sh Aleph-Alpha/Pharia-1-LLM-7B-control-hf 8 "Positive Negative Positive Negative"
sbatch run_generic.sh Aleph-Alpha/Pharia-1-LLM-7B-control-hf 8 "Positive Negative Negative Positive"
sbatch run_generic.sh Aleph-Alpha/Pharia-1-LLM-7B-control-hf 8 "Negative Positive Positive Negative"
sbatch run_generic.sh Aleph-Alpha/Pharia-1-LLM-7B-control-hf 8 "Negative Positive Negative Positive"
sbatch run_generic.sh Aleph-Alpha/Pharia-1-LLM-7B-control-hf 8 "Negative Negative Positive Positive"

# phi 3 with set
sbatch run_generic.sh Aleph-Alpha/Pharia-1-LLM-7B-control-hf 8 "Positive Positive Negative Negative" "--use_set_encoding"
sbatch run_generic.sh Aleph-Alpha/Pharia-1-LLM-7B-control-hf 8 "Positive Negative Positive Negative" "--use_set_encoding"
sbatch run_generic.sh Aleph-Alpha/Pharia-1-LLM-7B-control-hf 8 "Positive Negative Negative Positive" "--use_set_encoding"
sbatch run_generic.sh Aleph-Alpha/Pharia-1-LLM-7B-control-hf 8 "Negative Positive Positive Negative" "--use_set_encoding"
sbatch run_generic.sh Aleph-Alpha/Pharia-1-LLM-7B-control-hf 8 "Negative Positive Negative Positive" "--use_set_encoding"
sbatch run_generic.sh Aleph-Alpha/Pharia-1-LLM-7B-control-hf 8 "Negative Negative Positive Positive" "--use_set_encoding"

exit 0

# phi 3
sbatch run_generic.sh "microsoft/Phi-3-mini-4k-instruct" 8 "Positive Positive Negative Negative"
sbatch run_generic.sh "microsoft/Phi-3-mini-4k-instruct" 8 "Positive Negative Positive Negative"
sbatch run_generic.sh "microsoft/Phi-3-mini-4k-instruct" 8 "Positive Negative Negative Positive"
sbatch run_generic.sh "microsoft/Phi-3-mini-4k-instruct" 8 "Negative Positive Positive Negative"
sbatch run_generic.sh "microsoft/Phi-3-mini-4k-instruct" 8 "Negative Positive Negative Positive"
sbatch run_generic.sh "microsoft/Phi-3-mini-4k-instruct" 8 "Negative Negative Positive Positive"

# phi 3 with set
sbatch run_generic.sh "microsoft/Phi-3-mini-4k-instruct" 8 "Positive Positive Negative Negative" "--use_set_encoding"
sbatch run_generic.sh "microsoft/Phi-3-mini-4k-instruct" 8 "Positive Negative Positive Negative" "--use_set_encoding"
sbatch run_generic.sh "microsoft/Phi-3-mini-4k-instruct" 8 "Positive Negative Negative Positive" "--use_set_encoding"
sbatch run_generic.sh "microsoft/Phi-3-mini-4k-instruct" 8 "Negative Positive Positive Negative" "--use_set_encoding"
sbatch run_generic.sh "microsoft/Phi-3-mini-4k-instruct" 8 "Negative Positive Negative Positive" "--use_set_encoding"
sbatch run_generic.sh "microsoft/Phi-3-mini-4k-instruct" 8 "Negative Negative Positive Positive" "--use_set_encoding"

# Llama3
sbatch run_generic.sh "meta-llama/Meta-Llama-3-8B" 4 "Positive Positive Negative Negative" "--use_set_encoding"
sbatch run_generic.sh "meta-llama/Meta-Llama-3-8B" 4 "Positive Negative Positive Negative" "--use_set_encoding"
sbatch run_generic.sh "meta-llama/Meta-Llama-3-8B" 4 "Positive Negative Negative Positive" "--use_set_encoding"
sbatch run_generic.sh "meta-llama/Meta-Llama-3-8B" 4 "Negative Positive Positive Negative" "--use_set_encoding"
sbatch run_generic.sh "meta-llama/Meta-Llama-3-8B" 4 "Negative Positive Negative Positive" "--use_set_encoding"
sbatch run_generic.sh "meta-llama/Meta-Llama-3-8B" 4 "Negative Negative Positive Positive" "--use_set_encoding"

# Llama2
sbatch run_generic.sh "meta-llama/Llama-2-7b-hf" 8 "Positive Positive Negative Negative" "--use_set_encoding"
sbatch run_generic.sh "meta-llama/Llama-2-7b-hf" 8 "Positive Negative Positive Negative" "--use_set_encoding"
sbatch run_generic.sh "meta-llama/Llama-2-7b-hf" 8 "Positive Negative Negative Positive" "--use_set_encoding"
sbatch run_generic.sh "meta-llama/Llama-2-7b-hf" 8 "Negative Positive Positive Negative" "--use_set_encoding"
sbatch run_generic.sh "meta-llama/Llama-2-7b-hf" 8 "Negative Positive Negative Positive" "--use_set_encoding"
sbatch run_generic.sh "meta-llama/Llama-2-7b-hf" 8 "Negative Negative Positive Positive" "--use_set_encoding"

# Mistral
sbatch run_generic.sh "mistralai/Mistral-7B-v0.3" 8 "Positive Positive Negative Negative" "--use_set_encoding"
sbatch run_generic.sh "mistralai/Mistral-7B-v0.3" 8 "Positive Negative Positive Negative" "--use_set_encoding"
sbatch run_generic.sh "mistralai/Mistral-7B-v0.3" 8 "Positive Negative Negative Positive" "--use_set_encoding"
sbatch run_generic.sh "mistralai/Mistral-7B-v0.3" 8 "Negative Positive Positive Negative" "--use_set_encoding"
sbatch run_generic.sh "mistralai/Mistral-7B-v0.3" 8 "Negative Positive Negative Positive" "--use_set_encoding"
sbatch run_generic.sh "mistralai/Mistral-7B-v0.3" 8 "Negative Negative Positive Positive" "--use_set_encoding"

# Falcon
sbatch run_generic.sh "tiiuae/falcon-7b" 8 "Positive Positive Negative Negative" "--use_set_encoding"
sbatch run_generic.sh "tiiuae/falcon-7b" 8 "Positive Negative Positive Negative" "--use_set_encoding"
sbatch run_generic.sh "tiiuae/falcon-7b" 8 "Positive Negative Negative Positive" "--use_set_encoding"
sbatch run_generic.sh "tiiuae/falcon-7b" 8 "Negative Positive Positive Negative" "--use_set_encoding"
sbatch run_generic.sh "tiiuae/falcon-7b" 8 "Negative Positive Negative Positive" "--use_set_encoding"
sbatch run_generic.sh "tiiuae/falcon-7b" 8 "Negative Negative Positive Positive" "--use_set_encoding"

# gpt2
sbatch run_generic.sh "gpt2" 32 "Positive Positive Negative Negative" "--use_set_encoding"
sbatch run_generic.sh "gpt2" 32 "Positive Negative Positive Negative" "--use_set_encoding"
sbatch run_generic.sh "gpt2" 32 "Positive Negative Negative Positive" "--use_set_encoding"
sbatch run_generic.sh "gpt2" 32 "Negative Positive Positive Negative" "--use_set_encoding"
sbatch run_generic.sh "gpt2" 32 "Negative Positive Negative Positive" "--use_set_encoding"
sbatch run_generic.sh "gpt2" 32 "Negative Negative Positive Positive" "--use_set_encoding"

# Llama3-instruct
sbatch run_generic.sh "meta-llama/Meta-Llama-3-8B-instruct" 4 "Positive Positive Negative Negative" "--use_set_encoding"
sbatch run_generic.sh "meta-llama/Meta-Llama-3-8B-instruct" 4 "Positive Negative Positive Negative" "--use_set_encoding"
sbatch run_generic.sh "meta-llama/Meta-Llama-3-8B-instruct" 4 "Positive Negative Negative Positive" "--use_set_encoding"
sbatch run_generic.sh "meta-llama/Meta-Llama-3-8B-instruct" 4 "Negative Positive Positive Negative" "--use_set_encoding"
sbatch run_generic.sh "meta-llama/Meta-Llama-3-8B-instruct" 4 "Negative Positive Negative Positive" "--use_set_encoding"
sbatch run_generic.sh "meta-llama/Meta-Llama-3-8B-instruct" 4 "Negative Negative Positive Positive" "--use_set_encoding"

# Llama2-instruct
sbatch run_generic.sh "meta-llama/Llama-2-7b-chat-hf" 8 "Positive Positive Negative Negative" "--use_set_encoding"
sbatch run_generic.sh "meta-llama/Llama-2-7b-chat-hf" 8 "Positive Negative Positive Negative" "--use_set_encoding"
sbatch run_generic.sh "meta-llama/Llama-2-7b-chat-hf" 8 "Positive Negative Negative Positive" "--use_set_encoding"
sbatch run_generic.sh "meta-llama/Llama-2-7b-chat-hf" 8 "Negative Positive Positive Negative" "--use_set_encoding"
sbatch run_generic.sh "meta-llama/Llama-2-7b-chat-hf" 8 "Negative Positive Negative Positive" "--use_set_encoding"
sbatch run_generic.sh "meta-llama/Llama-2-7b-chat-hf" 8 "Negative Negative Positive Positive" "--use_set_encoding"

# Mistral-instruct
sbatch run_generic.sh "mistralai/Mistral-7B-Instruct-v0.3" 8 "Positive Positive Negative Negative" "--use_set_encoding"
sbatch run_generic.sh "mistralai/Mistral-7B-Instruct-v0.3" 8 "Positive Negative Positive Negative" "--use_set_encoding"
sbatch run_generic.sh "mistralai/Mistral-7B-Instruct-v0.3" 8 "Positive Negative Negative Positive" "--use_set_encoding"
sbatch run_generic.sh "mistralai/Mistral-7B-Instruct-v0.3" 8 "Negative Positive Positive Negative" "--use_set_encoding"
sbatch run_generic.sh "mistralai/Mistral-7B-Instruct-v0.3" 8 "Negative Positive Negative Positive" "--use_set_encoding"
sbatch run_generic.sh "mistralai/Mistral-7B-Instruct-v0.3" 8 "Negative Negative Positive Positive" "--use_set_encoding"

# Falcon-instruct
sbatch run_generic.sh "tiiuae/falcon-7b-instruct" 8 "Positive Positive Negative Negative" "--use_set_encoding"
sbatch run_generic.sh "tiiuae/falcon-7b-instruct" 8 "Positive Negative Positive Negative" "--use_set_encoding"
sbatch run_generic.sh "tiiuae/falcon-7b-instruct" 8 "Positive Negative Negative Positive" "--use_set_encoding"
sbatch run_generic.sh "tiiuae/falcon-7b-instruct" 8 "Negative Positive Positive Negative" "--use_set_encoding"
sbatch run_generic.sh "tiiuae/falcon-7b-instruct" 8 "Negative Positive Negative Positive" "--use_set_encoding"
sbatch run_generic.sh "tiiuae/falcon-7b-instruct" 8 "Negative Negative Positive Positive" "--use_set_encoding"





# Llama3
sbatch run_generic.sh "meta-llama/Meta-Llama-3-8B" 4 "Positive Positive Negative Negative"
sbatch run_generic.sh "meta-llama/Meta-Llama-3-8B" 4 "Positive Negative Positive Negative"
sbatch run_generic.sh "meta-llama/Meta-Llama-3-8B" 4 "Positive Negative Negative Positive"
sbatch run_generic.sh "meta-llama/Meta-Llama-3-8B" 4 "Negative Positive Positive Negative"
sbatch run_generic.sh "meta-llama/Meta-Llama-3-8B" 4 "Negative Positive Negative Positive"
sbatch run_generic.sh "meta-llama/Meta-Llama-3-8B" 4 "Negative Negative Positive Positive"

# Llama2
sbatch run_generic.sh "meta-llama/Llama-2-7b-hf" 8 "Positive Positive Negative Negative"
sbatch run_generic.sh "meta-llama/Llama-2-7b-hf" 8 "Positive Negative Positive Negative"
sbatch run_generic.sh "meta-llama/Llama-2-7b-hf" 8 "Positive Negative Negative Positive"
sbatch run_generic.sh "meta-llama/Llama-2-7b-hf" 8 "Negative Positive Positive Negative"
sbatch run_generic.sh "meta-llama/Llama-2-7b-hf" 8 "Negative Positive Negative Positive"
sbatch run_generic.sh "meta-llama/Llama-2-7b-hf" 8 "Negative Negative Positive Positive"

# Mistral
sbatch run_generic.sh "mistralai/Mistral-7B-v0.3" 8 "Positive Positive Negative Negative"
sbatch run_generic.sh "mistralai/Mistral-7B-v0.3" 8 "Positive Negative Positive Negative"
sbatch run_generic.sh "mistralai/Mistral-7B-v0.3" 8 "Positive Negative Negative Positive"
sbatch run_generic.sh "mistralai/Mistral-7B-v0.3" 8 "Negative Positive Positive Negative"
sbatch run_generic.sh "mistralai/Mistral-7B-v0.3" 8 "Negative Positive Negative Positive"
sbatch run_generic.sh "mistralai/Mistral-7B-v0.3" 8 "Negative Negative Positive Positive"

# Falcon
sbatch run_generic.sh "tiiuae/falcon-7b" 8 "Positive Positive Negative Negative"
sbatch run_generic.sh "tiiuae/falcon-7b" 8 "Positive Negative Positive Negative"
sbatch run_generic.sh "tiiuae/falcon-7b" 8 "Positive Negative Negative Positive"
sbatch run_generic.sh "tiiuae/falcon-7b" 8 "Negative Positive Positive Negative"
sbatch run_generic.sh "tiiuae/falcon-7b" 8 "Negative Positive Negative Positive"
sbatch run_generic.sh "tiiuae/falcon-7b" 8 "Negative Negative Positive Positive"

# gpt2
sbatch run_generic.sh "gpt2" 32 "Positive Positive Negative Negative"
sbatch run_generic.sh "gpt2" 32 "Positive Negative Positive Negative"
sbatch run_generic.sh "gpt2" 32 "Positive Negative Negative Positive"
sbatch run_generic.sh "gpt2" 32 "Negative Positive Positive Negative"
sbatch run_generic.sh "gpt2" 32 "Negative Positive Negative Positive"
sbatch run_generic.sh "gpt2" 32 "Negative Negative Positive Positive"

# Llama3-instruct
sbatch run_generic.sh "meta-llama/Meta-Llama-3-8B-instruct" 4 "Positive Positive Negative Negative"
sbatch run_generic.sh "meta-llama/Meta-Llama-3-8B-instruct" 4 "Positive Negative Positive Negative"
sbatch run_generic.sh "meta-llama/Meta-Llama-3-8B-instruct" 4 "Positive Negative Negative Positive"
sbatch run_generic.sh "meta-llama/Meta-Llama-3-8B-instruct" 4 "Negative Positive Positive Negative"
sbatch run_generic.sh "meta-llama/Meta-Llama-3-8B-instruct" 4 "Negative Positive Negative Positive"
sbatch run_generic.sh "meta-llama/Meta-Llama-3-8B-instruct" 4 "Negative Negative Positive Positive"

# Llama2-instruct
sbatch run_generic.sh "meta-llama/Llama-2-7b-chat-hf" 8 "Positive Positive Negative Negative"
sbatch run_generic.sh "meta-llama/Llama-2-7b-chat-hf" 8 "Positive Negative Positive Negative"
sbatch run_generic.sh "meta-llama/Llama-2-7b-chat-hf" 8 "Positive Negative Negative Positive"
sbatch run_generic.sh "meta-llama/Llama-2-7b-chat-hf" 8 "Negative Positive Positive Negative"
sbatch run_generic.sh "meta-llama/Llama-2-7b-chat-hf" 8 "Negative Positive Negative Positive"
sbatch run_generic.sh "meta-llama/Llama-2-7b-chat-hf" 8 "Negative Negative Positive Positive"

# Mistral-instruct
sbatch run_generic.sh "mistralai/Mistral-7B-Instruct-v0.3" 8 "Positive Positive Negative Negative"
sbatch run_generic.sh "mistralai/Mistral-7B-Instruct-v0.3" 8 "Positive Negative Positive Negative"
sbatch run_generic.sh "mistralai/Mistral-7B-Instruct-v0.3" 8 "Positive Negative Negative Positive"
sbatch run_generic.sh "mistralai/Mistral-7B-Instruct-v0.3" 8 "Negative Positive Positive Negative"
sbatch run_generic.sh "mistralai/Mistral-7B-Instruct-v0.3" 8 "Negative Positive Negative Positive"
sbatch run_generic.sh "mistralai/Mistral-7B-Instruct-v0.3" 8 "Negative Negative Positive Positive"

# Falcon-instruct
sbatch run_generic.sh "tiiuae/falcon-7b-instruct" 8 "Positive Positive Negative Negative"
sbatch run_generic.sh "tiiuae/falcon-7b-instruct" 8 "Positive Negative Positive Negative"
sbatch run_generic.sh "tiiuae/falcon-7b-instruct" 8 "Positive Negative Negative Positive"
sbatch run_generic.sh "tiiuae/falcon-7b-instruct" 8 "Negative Positive Positive Negative"
sbatch run_generic.sh "tiiuae/falcon-7b-instruct" 8 "Negative Positive Negative Positive"
sbatch run_generic.sh "tiiuae/falcon-7b-instruct" 8 "Negative Negative Positive Positive"