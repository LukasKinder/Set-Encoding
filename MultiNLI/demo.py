# Import libraries
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from set_encoding_utils import SetEncoder


# Define model and tokenizer
model_id = "microsoft/Phi-3-mini-4k-instruct"

# "meta-llama/Llama-2-7b-chat-hf", "meta-llama/Meta-Llama-3-8B-instruct", "gpt2","mistralai/Mistral-7B-Instruct-v0.3","tiiuae/falcon-7b-instruct", "microsoft/Phi-3-mini-4k-instruct", "Aleph-Alpha/Pharia-1-LLM-7B-control-hf"

tokenizer = AutoTokenizer.from_pretrained(model_id)
set_encoder = SetEncoder(tokenizer)

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

p1 = "<~start_set_marker~><~start_element_marker~> yes<~end_element_marker~><~start_element_marker~> apple<~end_element_marker~><~start_element_marker~> mo<~end_element_marker~><~end_set_marker~> end"

#SET(' yes', 'apple',' mo') ' end'
tokens1 = set_encoder(p1,device_for_output=model.device)
output1 = model(**tokens1)

print("\nLogits ' apple' in all prompts:")
print(f"Prompt 1: {output1.logits[0][4]}")
