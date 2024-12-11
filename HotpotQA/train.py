import argparse
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments ,Trainer 
import random
from tqdm import tqdm
import json

from set_encoding_utils import SetEncoder

"""
python train.py --save_path ./finetuned_models/phi3-set --max_tokens 2000 --num_epochs 3 --use_set_encoding True --model_id "microsoft/Phi-3-mini-4k-instruct" --learning_rate 1e-6 --accumulation_steps 4 --batch_size 1 
python train.py --save_path ./finetuned_models/phi3 --max_tokens 2000 --num_epochs 3 --use_set_encoding False --model_id "microsoft/Phi-3-mini-4k-instruct" --learning_rate 1e-6 --accumulation_steps 16 --batch_size 1
"""

# python train.py --save_path ./finetuned_models/phi3 --max_tokens 2000 --num_epochs 3 --use_set_encoding True --model_id "microsoft/Phi-3-mini-4k-instruct" --learning_rate 5e-6 --accumulation_steps 16 --batch_size 1 

# python train.py --save_path ./finetuned_models/llama3 --max_tokens 4000 --num_epochs 3 --use_set_encoding True --model_id "meta-llama/Meta-Llama-3-8B-instruct" --learning_rate 5e-6 --accumulation_steps 16 --batch_size 1 

class HotpotDataset(Dataset):
    def __init__(self, max_tokens, tokenizer, set_encoder, use_set_encoding ,data_path = "hotpot/hotpot_train_v1.1.json"):

        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.set_encoder = set_encoder
        self.use_set_encoding = use_set_encoding

        # Load the HotpotQA dataset
        print("Loading data...",end = "", flush = True)
        with open(data_path, "r") as file:
            data = json.load(file)
        print("Done")

        # 7405
        self.questions = []
        self.answers = []
        self.distracting_documents = []
        self.relevant_documents = []

        print("Processing Data...")
        for entry in tqdm(data):
            self.questions.append(entry['question'])
            a = entry['answer']
            if a == "no":
                a = "No"
            if a == "yes":
                a = "Yes"
            self.answers.append(a)
            
            supporting_facts = [x[0] for x in entry['supporting_facts']]
            self.relevant_documents.append([title + ":\n" + " ".join(sentences)  for title, sentences in entry['context'] if title in supporting_facts])
            self.distracting_documents.append([title + ":\n" + " ".join(sentences)  for title, sentences in entry['context'] if title not in supporting_facts])

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        all_documents = self.relevant_documents[idx]
        current_n_tokens = 6 + len(self.tokenizer.encode(all_documents[0]) + self.tokenizer.encode(all_documents[1]) + self.tokenizer.encode(self.questions[idx]) )
        
        n_tokens = random.randint(current_n_tokens,self.max_tokens)

        for i in range(0,100):
            if i < len(self.distracting_documents[idx]):
                new = self.distracting_documents[idx][i]
            else:

                random_one = self.distracting_documents[random.randint(0,len(self.questions) -1)]
                while len(random_one) == 0:
                    random_one = self.distracting_documents[random.randint(0,len(self.questions) -1)]
                new = random_one[random.randint(0, len(random_one) -1)]
            
            new_len = len(self.tokenizer.encode(new))
            if current_n_tokens + new_len > n_tokens:
                break
            current_n_tokens += new_len
            all_documents.append(new)

        #print(f" n_tokens = {current_n_tokens}, n_documents = {len(all_documents)}")

        random.shuffle(all_documents)
        prompt = "<~start_set_marker~>"
        for doc in all_documents:
            prompt += f"<~start_element_marker~>{doc}\n\n<~end_element_marker~>"

        prompt += f"<~end_set_marker~>\nQuestion: {self.questions[idx]}\nAnswer: {self.answers[idx]}"


        tokens = self.set_encoder(prompt)
        if not self.use_set_encoding:
            tokens["set_pos_encoding"]  = None
            tokens["set_attention_mask"] = None

        tokens["labels"] = tokens["input_ids"].clone()

        unsqueezed = dict()
        for k, v in tokens.items():
            if v == None:
                unsqueezed[k] = None
            else:
                unsqueezed[k] = v.squeeze(0)

        return unsqueezed

def main():
    parser = argparse.ArgumentParser(description="Finetune a language model on MMLU dataset.")
    # Must be specified
    parser.add_argument("--save_path", required=True, help="Path to save the finetuned model.")
    parser.add_argument("--use_set_encoding", required=True, help="Whether to use set encoding.")
    parser.add_argument("--model_id", required=True, help="Model ID for the pretrained model.")

    # Optionals
    parser.add_argument("--max_tokens", type=int, default=256, help="Maximum number of tokens.")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=4, help="Number of training epochs.")
    parser.add_argument("--warmup_steps", type=int, default=200, help="The number of warmup steps")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="Batch size.")

    args = parser.parse_args()
    use_set_encoding = args.use_set_encoding.lower() in ["true", "t", "1", "y", "yes"]
    n_accumulation_steps =  args.accumulation_steps

    print(f"""Running with:
        --save_path = {args.save_path}
        --use_set_encoding = {use_set_encoding}
        --model_id = {args.model_id}
        --max_tokens = {args.max_tokens}
        --learning_rate = {args.learning_rate}
        --num_epochs = {args.num_epochs}
        --warmup_steps = {args.warmup_steps}
        --batch_size = {args.batch_size}
        --accumulation_steps = {n_accumulation_steps}""")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    set_encoder = SetEncoder(tokenizer, custom_size=None)

    train_dataset = HotpotDataset(args.max_tokens, tokenizer, set_encoder = set_encoder,use_set_encoding = use_set_encoding)

    print(f"len train data: {len(train_dataset)}")
    print(f"n_warmup steps are: {args.warmup_steps}")
    print(f"total number of steps are: {int((len(train_dataset) / args.batch_size) * args.num_epochs)}")


    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, device_map="auto")
    model.config.pad_token_id = tokenizer.eos_token_id


    training_args = TrainingArguments(
        output_dir=args.save_path,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type= "constant_with_warmup",
        logging_dir='./logs',
        logging_steps=100,
        save_steps=120,
        gradient_accumulation_steps= n_accumulation_steps,
        optim="adafactor", # saves considerable memory, may lead to slower training
        bf16= True
        # fp16 = True # speed up, may cause instability
        # bf16= True # speed up, may reduce training effectivness because of rounding
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )

    trainer.train()

if __name__ == "__main__":
    main()
