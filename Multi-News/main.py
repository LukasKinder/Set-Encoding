import argparse
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import torch
import os
from torch.utils.data import DataLoader
from set_encoding_utils import SetEncoder
from dataloader import *
import gc

MAX_BATCHES = 99999

# Function to abbreviate model IDs
def abbreviate_model_id(model_id):
    if "llama-2" in model_id.lower():
        s = "Llama2-7b"
    elif "llama-3" in model_id.lower():
        s = "Llama3-8b"
    elif "mistral" in model_id.lower():
        s = "mistral"
    elif "falcon-7b" in model_id.lower():
        s = "falcon-7b"
    elif "phi-3" in model_id.lower():
        s = "phi-3"
    elif "pharia-1" in model_id.lower():
        s = "pharia-1"
    else:
        return model_id
    
    if "instruct" in model_id.lower() or "chat" in model_id.lower():
        s += "-instruct"
    return s

def args_to_file_name(args):
    name = "results/" 
    name += abbreviate_model_id(args.model_id) + "-" + f"maxT-{args.max_tokens}_max_TD-{args.max_tokens_per_document}"
    if args.use_set_encoding:
        name += "-set"
    return name + ".json"


def main(model_id, max_tokens, max_tokens_per_document,n_generate, use_set_encoding, verbose = False):

    # Load model and tokenizer on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "pharia" in model_id.lower():
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,trust_remote_code=True,load_in_4bit=True, device_map='auto')
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,load_in_4bit=True, device_map='auto')
    
    tokenizer = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)
    set_encoder = SetEncoder(tokenizer)

    # Initialize dataset and dataloader
    test_dataset = MultiNewsDataset(bucket_size= 10,max_total_tokens=max_tokens, max_tokens_document=max_tokens_per_document
                                    ,tokenizer=tokenizer,special_tokens_map=SPECIAL_TOKENS_MAP[model_id])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    prompts = []
    answers  = []
    gold_summaries = []
    n_tokens = []

    # Batch processing
    counter = 0
    for prompt, summary in tqdm(test_loader):
        counter +=1
        if counter == MAX_BATCHES:
            break

        inputs = set_encoder(prompt, device_for_output = device)

        if verbose:
            print(inputs["input_ids"].shape)
        n_tokens.append(inputs["input_ids"].shape[1])

        if not use_set_encoding:
            inputs["set_pos_encoding"]  = None
            inputs["set_attention_mask"] = None
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=n_generate
            )

        answer = tokenizer.decode(output[0],skip_special_tokens=False)

        if verbose:
            print("'''" + answer + "'''")
        answer = answer.split("Write a ~300 word summary about the given articles.")[1]

        prompts.append(prompt[0])
        answers.append(answer)
        gold_summaries.append(summary[0])

        #inputs = None
        #output = None
        #gc.collect()
        #torch.cuda.empty_cache()

    # Save results
    results = {
        "model": model_id,
        "set_encoding" : use_set_encoding,
        "max_tokens" : max_tokens,
        "max_tokens_per_document" : max_tokens_per_document,
        "n_tokens" : n_tokens,
        "prompts" : prompts,
        "answers" : answers,
        "gold_summaries" : gold_summaries
    }
    with open(save_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {save_file}")


if __name__ == "__main__":

    random.seed(42)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Natural Language Inference using few-shot learning with a pre-trained model.")
    parser.add_argument("--model_id", type=str, default="microsoft/Phi-3-mini-4k-instruct", help='Model ID for the transformer model, one of:  "meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-70b-chat-hf", "stabilityai/stablelm-tuned-alpha-7b", "tiiuae/falcon-40b-instruct"')
    parser.add_argument("--max_tokens", type=int, default=6000, help="Max number of tokens in the prompt (not including answer)")
    parser.add_argument("--max_tokens_per_document", type=int, default=1500, help="The maximum number of tokens per document")
    parser.add_argument("--n_generate", type=int, default=300, help="the number of tokens to generate")
    parser.add_argument("--use_set_encoding", action="store_true", help="Use set encoding.")

    args = parser.parse_args()

    save_file = args_to_file_name(args)
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    main(args.model_id, args.max_tokens, args.max_tokens_per_document,args.n_generate, args.use_set_encoding)


# "meta-llama/Llama-2-7b-chat-hf", "meta-llama/Meta-Llama-3-8B-instruct","gpt2","mistralai/Mistral-7B-Instruct-v0.3","tiiuae/falcon-7b-instruct", 
# "microsoft/Phi-3-mini-4k-instruct", "Aleph-Alpha/Pharia-1-LLM-7B-control-hf"