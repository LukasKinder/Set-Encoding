
# Import libraries
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from set_encoding_utils import SetEncoder, special_tokens_map


# This demo contains four example prompts, each involving a list of items encoded as a set. Specific markers indicate where each item begins, 
# and these markers are used solely by the set encoder. They are removed from the text after processing and do not appear as tokens input to the LLM.
# 
# In the first two examples, the LLM is tasked with reasoning about sets. With set encoding, it may struggle to determine the number of items 
# in the set or identify the first item. 
#
# The third and fourth examples are identical, except for the order of items in the set. Without set encoding, the Llama3 model 
# modifies its answers based on the order of items, demonstrating its order sensitivity.

# This flag can be used to switch Set-encoding on and off
USE_SET_ENCODING = True

# Possible Modedles adapted for set encoding: 
# "meta-llama/Llama-2-7b-chat-hf", "meta-llama/Meta-Llama-3-8B-instruct", "gpt2","mistralai/Mistral-7B-Instruct-v0.3","tiiuae/falcon-7b-instruct", "microsoft/Phi-3-mini-4k-instruct"
model_id = "meta-llama/Meta-Llama-3-8B-instruct"


# Define model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

set_encoder = SetEncoder(tokenizer)

prompts = [f"""{special_tokens_map[model_id]["start"]}Here is a list:
<~start_set_marker~><~start_element_marker~>-Apple
<~end_element_marker~><~start_element_marker~>-Potato
<~end_element_marker~><~start_element_marker~>-Tomato
<~end_element_marker~><~end_set_marker~>
What are the three items in the list above?{special_tokens_map[model_id]["end"]}""",

f"""{special_tokens_map[model_id]["start"]}Here is a list:
<~start_set_marker~><~start_element_marker~>-Apple
<~end_element_marker~><~start_element_marker~>-Potato
<~end_element_marker~><~start_element_marker~>-Tomato
<~end_element_marker~><~end_set_marker~>
What is the first item in the list?{special_tokens_map[model_id]["end"]}""",

f"""{special_tokens_map[model_id]["start"]}What would be the best present for a 7-year old?
<~start_set_marker~><~start_element_marker~>-Swing
<~end_element_marker~><~start_element_marker~>-Sweets
<~end_element_marker~><~start_element_marker~>-Puzzle
<~end_element_marker~><~end_set_marker~>
{special_tokens_map[model_id]["end"]}Of the given option the best present for a 7-year old is""",

f"""{special_tokens_map[model_id]["start"]}What would be the best present for a 7-year old?
<~start_set_marker~><~start_element_marker~>-Sweets
<~end_element_marker~><~start_element_marker~>-Puzzle
<~end_element_marker~><~start_element_marker~>-Swing
<~end_element_marker~><~end_set_marker~>
{special_tokens_map[model_id]["end"]}Of the given option the best present for a 7-year old is"""
]


# Tokenize the prompts
tokens = set_encoder(prompts, model.device)

if not USE_SET_ENCODING:
    tokens["set_pos_encoding"] = None
    tokens["set_attention_mask"] = None

# Run the model
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

torch.manual_seed(42)
with torch.no_grad():
    outputs = model.generate(
        **tokens,
        max_new_tokens=40,
        temperature = None,
        do_sample=False
    )

# Decode the generated tokens
answers = [tokenizer.decode(output,skip_special_tokens=True) for output in outputs]

# Print the answers
for i, answer in enumerate(answers):
    print("\n---------------------\n")
    print(f"'{answer}'\n")

