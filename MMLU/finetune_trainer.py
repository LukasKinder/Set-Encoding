import argparse
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

from evaluation import QuestionConstructor
from set_encoding_utils import SetEncoder
from mmlu_dataloader import MMLUDataset

# python finetune_trainer.py --save_path ./finetuned_models/falcon-normal --use_set_encoding False --model_id "tiiuae/falcon-7b-instruct" --learning_rate 5e-6 --accumulation_steps 16 --batch_size 4 --data_dir MMLU_data/val
# python finetune_trainer.py --save_path ./finetuned_models/mistral-normal --use_set_encoding False --model_id "mistralai/Mistral-7B-Instruct-v0.3" --learning_rate 1e-5 --accumulation_steps 16 --batch_size 4 --data_dir MMLU_data/val

class CustomMMLUDataset(Dataset):
    def __init__(self, data_dir, tokenizer, set_encoder, question_constructor, max_tokens, use_set_encoding, use_proportion = [0.25,0.25,0.25,0.25]):
        self.dataset = MMLUDataset(
            data_dir,
            filter_unshufflable=True,
            max_tokens=max_tokens,
            tokenizer=tokenizer,
            construct_question_function=question_constructor.construct_question,
            max_len_type="all_options",
            proportion= use_proportion
        )
        self.tokenizer = tokenizer
        self.map_letter_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        self.set_encoder = set_encoder
        self.use_set_encoding = use_set_encoding
        self.max_tokens = max_tokens

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        datapoint = self.dataset[idx]
        question = datapoint['question']
        correct_option = datapoint["options"][self.map_letter_index[datapoint['answer']]]
        question_with_answer = question + " " + correct_option

        if self.use_set_encoding:
            tokens = self.set_encoder(question_with_answer)
        else:
            tokens = self.tokenizer(question_with_answer, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_tokens)

        tokens["labels"] = tokens["input_ids"].clone()
        tokens = {k: v.squeeze(0) for k, v in tokens.items()}
        return tokens

def main():
    parser = argparse.ArgumentParser(description="Finetune a language model on MMLU dataset.")
    # Must be specified
    parser.add_argument("--save_path", required=True, help="Path to save the finetuned model.")
    parser.add_argument("--use_set_encoding", required=True, help="Whether to use set encoding.")
    parser.add_argument("--model_id", required=True, help="Model ID for the pretrained model.")

    # Optionals MMLU_data/val MMLU_data/auxiliary_train
    parser.add_argument("--data_dir", default='MMLU_data/auxiliary_train', help="Path to the training data directory.")
    parser.add_argument("--max_tokens", type=int, default=256, help="Maximum number of tokens.")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="Batch size.")
    parser.add_argument("--balance", default="0.25,0.25,0.25,0.25", help="The balance of options")

    args = parser.parse_args()
    use_set_encoding = args.use_set_encoding.lower() in ["true", "t", "1", "y", "yes"]
    n_accumulation_steps =  args.accumulation_steps
    balance = [float(x) for x in args.balance.split(",")]

    print(f"""Running with:
        --save_path = {args.save_path}
        --use_set_encoding = {use_set_encoding}
        --model_id = {args.model_id}
        --data_dir = {args.data_dir}
        --max_tokens = {args.max_tokens}
        --learning_rate = {args.learning_rate}
        --num_epochs = {args.num_epochs}
        --batch_size = {args.batch_size}
        --accumulation_steps = {n_accumulation_steps}
        --balance = {balance}""")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    set_encoder = SetEncoder(tokenizer, custom_size=args.max_tokens)

    q_c = QuestionConstructor(args.model_id, use_set_markers=use_set_encoding, use_letters=False, include_begin_of_text_tokens=False)

    train_dataset = CustomMMLUDataset(args.data_dir, tokenizer, set_encoder, q_c, args.max_tokens, use_set_encoding,use_proportion= balance)

    print(f"len train data: {len(train_dataset)}")
    print(f"total number of steps are: {int((len(train_dataset) / args.batch_size) * args.num_epochs)}")

    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, device_map="auto")
    # model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, device_map="auto") # about twice as fast and memory efficient
    model.config.pad_token_id = tokenizer.eos_token_id

    # Calculate save steps for 7 checkpoints including one at the end
    total_steps = int((len(train_dataset) / (args.batch_size * n_accumulation_steps) ) * args.num_epochs)
    save_steps = total_steps // 10  # This ensures 6 evenly spaced checkpoints + 1 at the end

    training_args = TrainingArguments(
        output_dir=args.save_path,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        lr_scheduler_type= "constant_with_warmup",
        logging_dir='./logs',
        logging_steps=10,
        save_steps=save_steps,
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
