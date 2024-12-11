If you have not already, update the supmudules with:

git submodule init

git submodule update

(Or alternativly use: git clone --recurse-submodules https://github.com/LukasKinder/Set-Encoding)



# Set-Encoding

We suggest using a viruel environment:

python -m venv my_venv

source my_venv/bin/activate


# Models

The models for which we implemented set encoding are "meta-llama/Llama-2-7b-chat-hf", "meta-llama/Meta-Llama-3-8B-instruct", "gpt2","mistralai/Mistral-7B-Instruct-v0.3","tiiuae/falcon-7b-instruct", "microsoft/Phi-3-mini-4k-instruct"

# Demo

python demo.py