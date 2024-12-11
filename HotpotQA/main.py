import argparse
import pandas as pd
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import torch
import os
from torch.utils.data import DataLoader
from set_encoding_utils import SetEncoder
import torch.nn.functional as F
from utils_device_map import get_device_map
import gc
from dataloader import HotpotDataset

MAX_BATCHES = 99999999999999

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
    name += abbreviate_model_id(args.model_id) + "-" + str(args.max_tokens)
    if args.use_set_encoding:
        name += "-set"
    return name + ".json"


def main(batch_size, model_id, save_file, use_set_encoding, max_tokens,tokenizer_path, verbose = False):

    # Load model and tokenizer on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #device_map = get_device_map(model_id,N_GPUS)
    if "pharia" in model_id.lower():
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,trust_remote_code=True,load_in_4bit=True, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,load_in_4bit=True, device_map="auto")

    #print(model.hf_device_map)
    #input("ök")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,trust_remote_code=True)
    set_encoder = SetEncoder(tokenizer)

    # Initialize dataset and dataloader
    test_dataset = HotpotDataset(max_tokens= max_tokens,tokenizer=tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_logprobs = [] # the logprobs to generate the correct answer
    max_correct = [] # If model would generate the correct answer with a temperature of 0
    total_number_of_tokens = [] # the number of tokens in the prompt

    # Batch processing
    counter = 0
    for prompts, answers in tqdm(test_loader):
        counter +=1
        if counter == MAX_BATCHES:
            break

        inputs = set_encoder(prompts, device_for_output = device)

        #print(inputs["input_ids"].shape)

        if not use_set_encoding:
            inputs["set_pos_encoding"]  = None
            inputs["set_attention_mask"] = None
        
        with torch.no_grad():
            outputs = model(**inputs)

        for input_ids, output, answer in zip(inputs["input_ids"],outputs.logits,answers):

            total_number_of_tokens.append(int(input_ids.shape[0]))

            answer_tokens_id = tokenizer.encode(" " + answer, add_special_tokens=False)
            if "Llama-2" in tokenizer_path or "Phi-3" in tokenizer_path or "Pharia-1" in tokenizer_path:
                answer_tokens_id.pop(0)

            len_answer_tokens = len(answer_tokens_id)
            print(len_answer_tokens)
            
            if verbose == True:
                print([(x,tokenizer.decode([x])) for x in answer_tokens_id])
                print([(x,tokenizer.decode([int(x)])) for x in input_ids[-len_answer_tokens:]])
                input()

            if list(input_ids[-len_answer_tokens:]) != answer_tokens_id:
                print([(x,tokenizer.decode([x])) for x in answer_tokens_id])
                print([(x,tokenizer.decode([int(x)])) for x in input_ids[-len_answer_tokens:]])
                exit("Error: tokens do not fit!")

            logprob = 1
            t0_correct = True
            for position,id in enumerate(answer_tokens_id):
                index = -len_answer_tokens + position-1

                softmax_output =  F.softmax(output[index], dim=0)
                max_prob, max_id  = torch.max(softmax_output,dim = 0)
                if max_id != id:
                    t0_correct = False

                if verbose:
                    print(f"max prob: {max_prob} for token '{max_id}' -> '{tokenizer.decode([max_id])}'")
                    print(f"answer prob: {softmax_output[id]} for token '{id}' -> '{tokenizer.decode([id])}'")
                    input()
                logprob *= float(softmax_output[id])
            all_logprobs.append(float(logprob))
            max_correct.append(t0_correct)

            inputs = None
            outputs = None
            gc.collect()
            torch.cuda.empty_cache()
            #input("all cleared (?)")

    # Save results
    results = {
        "model": model_id,
        "set_encoding" : use_set_encoding,
        "max_tokens" : max_tokens,
        "n_tokens" : total_number_of_tokens,
        "logprobs" : all_logprobs,
        "correct_t0" : max_correct
    }
    with open(save_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {save_file}")


if __name__ == "__main__":

    random.seed(42)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Natural Language Inference using few-shot learning with a pre-trained model.")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-chat-hf", help='Model ID for the transformer model, one of:  "meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-70b-chat-hf", "stabilityai/stablelm-tuned-alpha-7b", "tiiuae/falcon-40b-instruct"')
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference.")
    parser.add_argument("--max_tokens", type=int, default=17000, help="Max number of tokens (add more documents)")
    parser.add_argument("--use_set_encoding", action="store_true", help="Use set encoding.")
    parser.add_argument('--tokenizer_path', type=str, default=None, help="the path of the tokenizer, if different to the model id")

    args = parser.parse_args()

    if args.tokenizer_path == None:
        args.tokenizer_path = args.model_id

    save_file = args_to_file_name(args)
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    main(args.batch_size, args.model_id, save_file, args.use_set_encoding, args.max_tokens, args.tokenizer_path)



"""
python main.py --model_id "finetuned_models/llama3-set/checkpoint-480"  --batch_size 1 --use_set_encoding --max_tokens 17000 --tokenizer_path meta-llama/Meta-Llama-3-8B-instruct 
python main.py --model_id "finetuned_models/llama3/checkpoint-480"  --batch_size 1 --max_tokens 17000 --tokenizer_path meta-llama/Meta-Llama-3-8B-instruct 
python main.py --model_id "finetuned_models/llama3-set/checkpoint-1080"  --batch_size 1 --use_set_encoding --max_tokens 17000 --tokenizer_path meta-llama/Meta-Llama-3-8B-instruct 
python main.py --model_id "finetuned_models/llama3/checkpoint-1080"  --batch_size 1 --max_tokens 17000 --tokenizer_path meta-llama/Meta-Llama-3-8B-instruct 
python main.py --model_id "finetuned_models/llama3-set/checkpoint-1680"  --batch_size 1 --use_set_encoding --max_tokens 17000 --tokenizer_path meta-llama/Meta-Llama-3-8B-instruct 
python main.py --model_id "finetuned_models/llama3/checkpoint-1680"  --batch_size 1 --max_tokens 17000 --tokenizer_path meta-llama/Meta-Llama-3-8B-instruct 


python main.py --model_id "finetuned_models/phi3-set/checkpoint-120"  --batch_size 1 --use_set_encoding --max_tokens 6500 --tokenizer_path microsoft/Phi-3-mini-4k-instruct 
python main.py --model_id "finetuned_models/phi3/checkpoint-120"  --batch_size 1 --max_tokens 6500 --tokenizer_path microsoft/Phi-3-mini-4k-instruct

python main.py --model_id tiiuae/falcon-7b --batch_size 1 --use_set_encoding --max_tokens 6500
python main.py --model_id tiiuae/falcon-7b --batch_size 1  --max_tokens 6500


python main.py --model_id meta-llama/Meta-Llama-3-8B-instruct --batch_size 1 --use_set_encoding --max_tokens 17000 --tokenizer_path 
python main.py --model_id meta-llama/Meta-Llama-3-8B-instruct --batch_size 1 --max_tokens 17000

python main.py --model_id Aleph-Alpha/Pharia-1-LLM-7B-control-hf --batch_size 1 --use_set_encoding --max_tokens 1000
python main.py --model_id Aleph-Alpha/Pharia-1-LLM-7B-control-hf --batch_size 1 --max_tokens 13000
"""