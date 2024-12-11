# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Prepare prediction jsonl with field `pred` .
dataset jsonl:
{
    "index" int,
    "input": str,
    "outputs": [str],
}

prediction jsonl: 
{
    "index" int,
    "input": str,
    "outputs": [str],
    "pred": str,
}
"""

import argparse
import json
import yaml
import os
import sys
import threading
import importlib
import math
import time
from tqdm import tqdm
from pathlib import Path
import traceback

from nemo.utils.data_utils import DataStoreObject

#Lukas Imports
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from set_encoding_utils import SetEncoder

def read_manifest(manifest):
    manifest = DataStoreObject(str(manifest))
    data = []
    try:
        f = open(manifest.get(), 'r', encoding='utf-8')
    except:
        raise Exception(f"Manifest file could not be opened: {manifest}")

    errors = []
    for line in f.readlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            errors.append(line)
            continue
        data.append(item)
    f.close()
    return data

SERVER_TYPES = (
    'trtllm',
    'vllm',
    'sglang',
    'openai',
    'gemini',
    'hf',
    'mamba',
)


class ServerAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        namespace.server_type = values


parser = argparse.ArgumentParser()
# Data
parser.add_argument("--data_dir", type=Path, required=True, help='path to load the dataset jsonl files')
parser.add_argument("--save_dir", type=Path, required=True, help='path to save the prediction jsonl files')
parser.add_argument("--benchmark", type=str, default='synthetic', help='Options: [synthetic]')
parser.add_argument("--task", type=str, required=True, help='Options: tasks in benchmark')
parser.add_argument("--subset", type=str, default='validation', help='Options: validation or test')
parser.add_argument("--chunk_idx", type=int, default=0, help='index of current split chunk')
parser.add_argument("--chunk_amount", type=int, default=1, help='size of split chunk')

# Server
parser.add_argument("--server_type", default='nemo', action=ServerAction, choices=SERVER_TYPES)
parser.add_argument("--server_host", type=str, default='127.0.0.1')
parser.add_argument("--server_port", type=str, default='5000')
parser.add_argument("--ssh_server", type=str)
parser.add_argument("--ssh_key_path", type=str)
parser.add_argument("--model_name_or_path", type=str, default='gpt-3.5-turbo', 
                    help='supported models from OpenAI or HF (provide a key or a local path to the checkpoint)')

# Inference
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_k", type=int, default=32)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--random_seed", type=int, default=0)
parser.add_argument("--stop_words", type=str, default='')
parser.add_argument("--sliding_window_size", type=int)
parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=1)

#import sys
#print(sys.argv)
#input("This is the input")

args = parser.parse_args()
args.stop_words = list(filter(None, args.stop_words.split(',')))
if args.server_type == 'hf' or args.server_type == 'gemini':
    args.threads = 1

def split_text_into_elements(tokenizer, text, element_size=100):
    # Tokenize the text
    tokens = tokenizer.tokenize(text)
    
    # Split tokens into chunks of the specified size
    chunks = [tokens[i:i + element_size] for i in range(0, len(tokens), element_size)]
    
    # Convert each chunk of tokens back into a string
    chunked_texts = [tokenizer.convert_tokens_to_string(chunk) for chunk in chunks]
    
    return chunked_texts

def add_set_markers(prompt,task, tokenizer):

    element_size = 1000

    # set start right after this string
    start_set_pos = { "niah_single_1" : "I will quiz you about the number afterwards.\n"
                     ,"niah_single_2" : "I will quiz you about the number afterwards.\n"
                     ,"niah_single_3" : "I will quiz you about the uuid afterwards.\n"
                     ,"niah_multikey_1" : "I will quiz you about the number afterwards.\n"
                     ,"niah_multikey_2" : "I will quiz you about the number afterwards.\n"
                     ,"niah_multikey_3" : "I will quiz you about the uuid afterwards.\n"
                     ,"niah_multiquery" : "I will quiz you about the numbers afterwards.\n"
                     ,"niah_multivalue" : "I will quiz you about the numbers afterwards.\n"
                     ,"vt" : "Memorize and track the chain(s) of variable assignment hidden in the following text."
                     ,"cwe" : "Memorize the ones that appear most often.\n"
                     ,"fwe" : "Find the three most frequently appeared coded words."
                     ,"qa_1" : "The following are given documents.\n\n"
                     ,"qa_2" : "The following are given documents.\n\n"
    }

    # Set should end right before this string:
    end_set_pos = {   "niah_single_1" : "\nWhat is the special magic number for"
                     ,"niah_single_2" : "\nWhat is the special magic number for"
                     ,"niah_single_3" : "\nWhat is the special magic uuid for"
                     ,"niah_multikey_1" : "\nWhat is the special magic number for"
                     ,"niah_multikey_2" : "\nWhat is the special magic number for"
                     ,"niah_multikey_3" : "\nWhat is the special magic uuid for"
                     ,"niah_multiquery" : "\nWhat are all the special magic numbers for"
                     ,"niah_multivalue" : "\nWhat are all the special magic numbers for"
                     ,"vt" : "\n\nQuestion: Find all variables that are assigned the value"
                     ,"cwe" : "\nQuestion: What are the 10 most common words in the above list?"
                     ,"fwe" : "\nQuestion: Do not provide any explanation."
                     ,"qa_1" : "\n\nAnswer the question based on the given documents."
                     ,"qa_2" : "\n\nAnswer the question based on the given documents."
    }

    start_position = prompt.find(start_set_pos[task]) + len(start_set_pos[task])
    end_position = prompt.rfind(end_set_pos[task])
    
    pre_set_part = prompt[:start_position]
    middle_part = prompt[start_position:end_position]
    post_set_part = prompt[end_position:]

    new_middle_part = ""
    for chunk in split_text_into_elements(tokenizer,middle_part,element_size = element_size):
        new_middle_part += "<~start_element_marker~>" + chunk + "<~end_element_marker~>"

    final_prompt = pre_set_part + "<~start_set_marker~>" + new_middle_part + "<~end_set_marker~>" + post_set_part

    return final_prompt


def run_set_encoding_llm(model,tokenizer,set_encoder,tokens_to_generate,prompts,task):
    all_answers = []
    for prompt in prompts:
        prompt = add_set_markers(prompt,task,set_encoder.tokenizer)

        tokens = set_encoder(prompt,device_for_output=model.device)

        use_set_encoding = True
        if not use_set_encoding:
            tokens["set_pos_encoding"]  = None
            tokens["set_attention_mask"] = None

        with torch.no_grad():
            outputs = model.generate(
                **tokens,
                max_new_tokens=tokens_to_generate
        )
            
        answer = tokenizer.decode(outputs[0],skip_special_tokens=True)
        answer  = answer.split("assistant")[-1]

        print(tokens["input_ids"].shape)
        print(f"answer is: '''{answer}'''")
        all_answers.append({"text" : answer})
    return all_answers

def main():
    start_time = time.time()
    
    curr_folder = os.path.dirname(os.path.abspath(__file__))
    
    try:
        sys.path.append(os.path.dirname(curr_folder))
        module = importlib.import_module(f"data.{args.benchmark}.constants")
    except ImportError:
        print(f"Module data.{args.benchmark}.constants not found.")

    tasks_base = module.TASKS
    with open(os.path.join(curr_folder, f"../{args.benchmark}.yaml"), "r") as f:
        tasks_customized = yaml.safe_load(f)

    if args.task not in tasks_customized:
        raise ValueError(f'{args.task} is not found in config_tasks.yaml')
        
    config = tasks_customized.get(args.task)
    config.update(tasks_base[config['task']])

    task_file = args.data_dir / args.task / f'{args.subset}.jsonl'
    
    if args.chunk_amount > 1:
        pred_file = args.save_dir / f'{args.task}-{args.chunk_idx}.jsonl'
    else:
        pred_file = args.save_dir / f'{args.task}.jsonl'
        
    print(f'Predict {args.task} \nfrom {task_file}\nto {pred_file}')
    pred_file.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    if os.path.exists(pred_file):
        pred_index = [sample['index'] for sample in read_manifest(pred_file)]
        data = [sample for sample in read_manifest(task_file) if sample['index'] not in pred_index]
    else:
        data = read_manifest(task_file)

    # Load api
    #llm = get_llm(config['tokens_to_generate'])

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-instruct")
    if "131072" in str(args.data_dir):
        custom_device_map = {'model.embed_tokens': 0
                      , 'model.layers.0': 1, 'model.layers.1': 1, 'model.layers.2': 1, 'model.layers.3': 1, 'model.layers.4': 1, 'model.layers.5': 1, 'model.layers.6': 1, 'model.layers.7': 1, 'model.layers.8': 1, 'model.layers.9': 1, 'model.layers.10': 1
                      , 'model.layers.11': 2, 'model.layers.12': 2, 'model.layers.13': 2, 'model.layers.14': 2, 'model.layers.15': 2, 'model.layers.16': 2, 'model.layers.17': 2, 'model.layers.18': 2, 'model.layers.19': 2, 'model.layers.20': 2, 'model.layers.21': 2
                      , 'model.layers.22': 3, 'model.layers.23': 3, 'model.layers.24': 3, 'model.layers.25': 3, 'model.layers.26': 3, 'model.layers.27': 3, 'model.layers.28': 3, 'model.layers.29': 3, 'model.layers.30': 3, 'model.layers.31': 3, 'model.norm': 3, 'lm_head': 3}
    else:
        custom_device_map = "auto"
    print(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16, device_map=custom_device_map)
    print(model.hf_device_map)

    set_encoder = SetEncoder(tokenizer)
    tokens_to_generate = config['tokens_to_generate']

    def get_output(idx_list, index_list, input_list, outputs_list, others_list, truncation_list, length_list,task):


        pred_list = run_set_encoding_llm(model,tokenizer,set_encoder,tokens_to_generate,prompts=input_list,task = task)

        zipped_iter = zip(pred_list, idx_list, index_list, input_list,
                          outputs_list, others_list, truncation_list, length_list)

        for pred, idx, index, input, outputs, others, truncation, length in zipped_iter:
            if isinstance(pred['text'], str):
                pred_text = pred['text']
            elif len(pred['text']) > 0:
                pred_text = pred['text'][0]
            else:
                pred_text = ''

            outputs_parallel[idx] = {
                'index': index,
                'pred': pred_text,
                'input': input,
                'outputs': outputs,
                'others': others,
                'truncation': truncation,
                'length': length,
            }

    threads = []
    outputs_parallel = [{} for _ in range(len(data))]

    batched_data = []
    batch = []
    for idx, data_point in enumerate(data):
        data_point['idx'] = idx

        if len(batch) >= args.batch_size:
            batched_data.append(batch)
            batch = []

        batch.append(data_point)

    if len(batch):
        batched_data.append(batch)

    # setting buffering=1 to force to dump the output after every line, so that we can see intermediate generations
    with open(pred_file, 'at', encoding="utf-8", buffering=1) as fout:
        # the data is processed sequentially, so we can store the start and end of current processing window
        start_idx = 0  # window: [start_idx, end_idx]

        for batch_idx, batch in tqdm(enumerate(batched_data), total=len(batched_data)):
            idx_list = [data_point['idx'] for data_point in batch]
            end_idx = idx_list[-1]  # the data in a batch is ordered

            thread = threading.Thread(
                target=get_output,
                kwargs=dict(
                    idx_list=idx_list,
                    index_list=[data_point['index'] for data_point in batch],
                    input_list=[data_point['input'] for data_point in batch],
                    outputs_list=[data_point['outputs'] for data_point in batch],
                    others_list=[data_point.get('others', {}) for data_point in batch],
                    truncation_list=[data_point.get('truncation', -1) for data_point in batch],
                    length_list=[data_point.get('length', -1) for data_point in batch],
                    task = args.task,
                ),
            )
            thread.start()
            threads.append(thread)

            is_last_batch = (batch_idx == len(batched_data) - 1)

            if (len(threads) == args.threads) or is_last_batch:
                for thread in threads:
                    thread.join()
                threads = []

                # dump the results in current processing window on disk
                for idx in range(start_idx, end_idx + 1):
                    if len(outputs_parallel[idx]) > 0:
                        fout.write(json.dumps(outputs_parallel[idx]) + '\n')

                start_idx = end_idx + 1

    print(f"Used time: {round((time.time() - start_time) / 60, 1)} minutes")


if __name__ == '__main__':
    main()
