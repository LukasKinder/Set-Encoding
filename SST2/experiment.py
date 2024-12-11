
# Import libraries
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# Define model and tokenizer
model_id = "tiiuae/falcon-7b"
# "meta-llama/Llama-2-7b-chat-hf", "meta-llama/Meta-Llama-3-8B-instruct","mistralai/Mistral-7B-Instruct-v0.3","tiiuae/falcon-7b-instruct"

# Not instruction tunes
# "meta-llama/Llama-2-7b-hf", "meta-llama/Meta-Llama-3-8B", "gpt2","mistralai/Mistral-7B-v0.3","tiiuae/falcon-7b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")


prompts = ["""Review: a uniquely sensual metaphorical dramatization of sexual obsession that spends a bit too much time on its fairly ludicrous plot .
Sentiment: Positive

Review: the rest of the film ... is dudsville .
Sentiment: Negative

Review: despite bearing the paramount imprint , it 's a bargain-basement european pickup .
Sentiment: Negative

Review: -lrb- `` take care of my cat '' -rrb- is an honestly nice little film that takes us on an examination of young adult life in urban south korea through the hearts and minds of the five principals .
Sentiment: Positive

Review: i loved it !
Sentiment:""",
"""Some filler: ßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßßß"""]


model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

# Tokenize the prompts
tokens = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

torch.manual_seed(42)
with torch.no_grad():
    outputs = model.generate(
        **tokens,
        max_new_tokens=1
    )

# Decode the generated tokens
answers = [tokenizer.decode(output,skip_special_tokens=True) for output in outputs]

print(answers[0])
input("kjfhjäw")

with torch.no_grad():
    outputs = model(**tokens)

# Get logits for the next token prediction
next_token_logits = outputs.logits[:, -1, :]
max_X = -99999
token_X = 0
for i_X in range(next_token_logits.shape[1]):
    if next_token_logits[0, i_X] > max_X:
        max_X = next_token_logits[0, i_X]
        token_X = i_X
print(f"best fitting token is: '''{tokenizer.decode([token_X])}'''")


exit(0)

# Print the answers
for i, answer in enumerate(answers):
    print("\n---------------------\n")
    print(f"'{answer}'\n")
    input()

