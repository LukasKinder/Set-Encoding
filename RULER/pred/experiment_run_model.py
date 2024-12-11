
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


model = AutoModelForCausalLM.from_pretrained('finetuned_models/Llama3-8klearning/checkpoint-3250', torch_dtype=torch.bfloat16, device_map="auto")
input("All good here")