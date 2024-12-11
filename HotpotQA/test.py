from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import torch
from utils_device_map import get_device_map

N_GPUS = 8

MAX_MEMORY = {i: '46GB' for i in range(N_GPUS)}

# model_id = "meta-llama/Meta-Llama-3-8B-instruct" # max tokens with 4 small gpus seems ~24k
# model_id = "tiiuae/falcon-7b" # max tokens with 4 small gpus seems 6.5k
# model_id = "mistralai/Mistral-7B-Instruct-v0.3" # ~27 k seem duable
# model_id = "microsoft/Phi-3-mini-4k-instruct" # ~7k
# model_id = "Aleph-Alpha/Pharia-1-LLM-7B-control-hf" # 6k
# model_id = "meta-llama/Llama-2-7b-hf"
model_id = "finetuned_models/phi3-set/checkpoint-10"
# model_id = "microsoft/Phi-3-mini-4k-instruct"

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct",trust_remote_code=True)

prompt = "How to make mayo?"

inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs)

print(output)
exit(0)

# Generate output with the model, without computing gradients
with torch.no_grad():
    output = model.generate(
        input_ids=inputs["input_ids"].to(model.device),
        attention_mask=inputs["attention_mask"].to(model.device),
        max_new_tokens=50
    )

print(output)

# Decode the output tokens into text
answer = tokenizer.decode(output[0], skip_special_tokens=True)

print(answer)
