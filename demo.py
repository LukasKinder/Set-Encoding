
# Import libraries
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from set_encoding_utils import SetEncoder

USE_SET_ENCODING =True

# Define model and tokenizer
model_id = "microsoft/Phi-3-mini-4k-instruct"

# Possible Modedles adapted for set encoding: 
# "meta-llama/Llama-2-7b-chat-hf", "meta-llama/Meta-Llama-3-8B-instruct", "gpt2","mistralai/Mistral-7B-Instruct-v0.3","tiiuae/falcon-7b-instruct", "microsoft/Phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

set_encoder = SetEncoder(tokenizer)

special_tokens_map = {
    "meta-llama/Llama-2-7b-chat-hf" : {"start" : "[INST]", "end" : "[/INST]"},
    "meta-llama/Meta-Llama-3-8B-instruct" : {"start" : "<|start_header_id|>user<|end_header_id|>\n\n", 
                                             "end" : "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"},
    "gpt2" : {"start" : "", "end" : "\n\n"},
    "mistralai/Mistral-7B-Instruct-v0.3" : {"start" : "[INST]",
                                             "end" : "[/INST]"},
    "tiiuae/falcon-7b-instruct" : {"start" : ">>QUESTION<<",
                                   "end" : ">>ANSWER<<"},
    "microsoft/Phi-3-mini-4k-instruct" : {"start" : "<|user|>\n",
                                          "end" : "<|end|>\n<|assistant|>"},
    "Aleph-Alpha/Pharia-1-LLM-7B-control-hf" : {"start" : "<|start_header_id|>user<|end_header_id|>\n\n", 
                                                "end" : "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"}                      
}

#order = ["Apple","Potato","Tomato"]
order = ["Potato","Tomato","Apple"]

prompts = [f"""{special_tokens_map[model_id]["start"]}Here is a list:
<~start_set_marker~><~start_element_marker~>-{order[0]}
<~end_element_marker~><~start_element_marker~>-{order[1]}
<~end_element_marker~><~start_element_marker~>-{order[2]}
<~end_element_marker~><~end_set_marker~>
Repeat the List.{special_tokens_map[model_id]["end"]}""",

f"""{special_tokens_map[model_id]["start"]}Here is a list:
<~start_set_marker~><~start_element_marker~>-{order[0]}
<~end_element_marker~><~start_element_marker~>-{order[1]}
<~end_element_marker~><~start_element_marker~>-{order[2]}
<~end_element_marker~><~end_set_marker~>
What is the first item in the list?{special_tokens_map[model_id]["end"]}""",

f"""{special_tokens_map[model_id]["start"]}Here is a list
<~start_set_marker~><~start_element_marker~>-{order[0]}
<~end_element_marker~><~start_element_marker~>-{order[1]}
<~end_element_marker~><~start_element_marker~>-{order[2]}
<~end_element_marker~><~end_set_marker~>
What is the second item in the list?{special_tokens_map[model_id]["end"]}""",
]

# Tokenize the prompts
tokens = set_encoder(prompts, model.device)

if not USE_SET_ENCODING:
    tokens["set_pos_encoding"]  = None
    tokens["set_attention_mask"] = None

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

torch.manual_seed(42)
with torch.no_grad():
    outputs = model.generate(
        **tokens,
        max_new_tokens=40
    )

# Decode the generated tokens
answers = [tokenizer.decode(output,skip_special_tokens=True) for output in outputs]

# Print the answers
for i, answer in enumerate(answers):
    print("\n---------------------\n")
    print(f"'{answer}'\n")

