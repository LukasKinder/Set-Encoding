## Set-Encoding

# Cloning the Repository

If you haven’t already, initialize and update the submodules:

```bash
git submodule init  
git submodule update  
```

Alternatively, you can clone the repository with its submodules directly:

```bash
git clone --recurse-submodules https://github.com/LukasKinder/Set-Encoding  
```

# Virtual Environment Setup

We recommend using a virtual environment:

```bash
python -m venv my_venv  
source my_venv/bin/activate  
pip install -r requirements_demo.txt  
```

The repository includes two lists of dependencies:

 - *requirements_demo.txt*: Contains the requirements needed to run the demo.
 - *requirements_all_experiments.txt*: Includes all dependencies for running the experiments.

**Important:** Do not install transformers directly. Use the modified version, SetTransformers.


# Supported Models

The following models are implemented with set encoding:

 - *meta-llama/Llama-2-7b-chat-hf*
 - *meta-llama/Meta-Llama-3-8B-instruct*
 - *gpt2*
 - *mistralai/Mistral-7B-Instruct-v0.3*
 - *tiiuae/falcon-7b-instruct*
 - *microsoft/Phi-3-mini-4k-instruct*

# DEMO

Run the demo script:
```bash
python demo.py  
```

This demo performs a simple experiment where a model is tasked with reasoning about lists encoded as a set.

# Experiments

The repository includes six experiment directories:

1. **HotpotQA**     (context window extension)
2. **MMLU**         (Positional Debiasing)
3. **Multi-News**   (context window extension)
4. **MultiNLI**     (Positional Debiasing)
5. **RULER**        (context window extension)
6. **SST**          (Positional Debiasing)