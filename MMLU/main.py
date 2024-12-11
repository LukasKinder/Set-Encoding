import argparse
import os
import json
from datetime import datetime
from evaluation import MMLU_Evaluator
import torch

# python main.py Aleph-Alpha/Pharia-1-LLM-7B-control-hf --batch_size 8 --max_batches 99999 --use_set_encoding --eval_type option --data_dir MMLU_data/filtered_test --first_tokens_differ

# python main.py "finetuned_models/mistral-set/checkpoint-564" --batch_size 32 --use_set_encoding --max_batches 99999 --eval_type "option" --tokenizer_path "mistralai/Mistral-7B-Instruct-v0.3" --first_tokens_differ

# python main.py meta-llama/Llama-2-7b-chat-hf --batch_size 32 --max_batches 99999 --eval_type option --tokenizer_path meta-llama/Llama-2-7b-chat-hf --first_tokens_differ
# python main.py "gpt2" --first_tokens_differ
# python main.py "gpt2" --tokenizer_path "gpt2" --eval_type "letter"
# python main.py "meta-llama/Meta-Llama-3-8B-instruct" --tokenizer_path "meta-llama/Meta-Llama-3-8B-instruct" --eval_type "letter"

#Examples:
# python main.py "meta-llama/Meta-Llama-3-8B-instruct" --use_set_encoding
# python main.py "meta-llama/Meta-Llama-3-8B-instruct" --use_set_encoding --batch_size 32 --eval_type "option" --first_tokens_differ --max_tokens 256
# python main.py "finetuned_models/llama-2-set/checkpoint-20" --batch_size 32 --use_set_encoding --max_batches 99999 --eval_type "option" --tokenizer_path "meta-llama/Llama-2-7b-chat-hf" --first_tokens_differ
# python main.py "finetuned_models/llama-3-set/checkpoint-20" --batch_size 32 --use_set_encoding --max_batches 99999 --eval_type "option" --tokenizer_path "meta-llama/Meta-Llama-3-8B-instruct" --first_tokens_differ

def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description='runs an evaluation with the MMLU dataset.')

    #required
    parser.add_argument('model_id', type=str, help='The model id for the LLM and tokenizer')

    #optionals
    parser.add_argument('--data_dir', type=str, default='MMLU_data/filtered_test', help='The path to the folder with csv filder for the MMLU dataset.')
    parser.add_argument('--eval_type', type=str, default='option', help="('letter', 'option' or 'free') The type of evaluation.")
    parser.add_argument('--use_set_encoding', action='store_true', help='If set_encoding should be used') # False by default!
    parser.add_argument('--batch_size', type=int, default = 64, help='The batch size') # may be too large for most runs
    parser.add_argument('--max_batches', type=int, default = 9999999, help='The maximum amount of batches to use')
    parser.add_argument('--tokenizer_path', type=str, default=None, help="the path of the tokenizer, if different to the model id")
    parser.add_argument('--first_tokens_differ', action='store_true', help='If only question should be used in which the options start with different tokens(makes the evaluation easier/quicker)') # False by default!


    # Parse the arguments
    args = parser.parse_args()
    assert args.eval_type in ["letter", "option","free"]
    if args.eval_type == 'option':
        assert args.batch_size % 4 == 0
        
    if args.tokenizer_path == None:
        args.tokenizer_path = args.model_id

    return args

def save_results(results, args):
    # Create the results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')

    # Get the current time for a unique filename
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.use_set_encoding:
        filename = f"results/{args.model_id.replace('/', '_')}_SET_{args.eval_type.upper()}_{current_time}.txt"
    else:
        filename = f"results/{args.model_id.replace('/', '_')}_NORMAL_{args.eval_type.upper()}_{current_time}.txt"

    # Combine results and args into a single dictionary
    combined_data = {
        "args": vars(args),
        "results": results
    }

    # Save the combined data to a text file
    with open(filename, 'w') as f:
        json.dump(combined_data, f, indent=4)

    print(f"Results saved to {filename}")

def main():

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    args = parse_args()
    print(args)

    evaluator = MMLU_Evaluator(**vars(args))
    results  = evaluator()

    save_results(results,args)

if __name__ == "__main__":
    main()

