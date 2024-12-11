import argparse
import pandas as pd
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import json
from tqdm import tqdm
import torch
import os
from torch.utils.data import Dataset, DataLoader
from set_encoding_utils import SetEncoder

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

# SST-2 Dataset Class
class SST2Dataset(Dataset):
    def __init__(self, sentences, labels, train_data, order):
        self.sentences = sentences
        self.labels = labels
        self.train_pos = train_data[train_data['label'] == 1]
        self.train_neg = train_data[train_data['label'] == 0]
        self.index_pos = 0
        self.index_neg = 0
        self.order = order

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        prompt = self.build_prompt(sentence)
        return prompt, label

    def get_few_shot_examples(self):
        """Get random few-shot examples based on the specified order."""
        examples = []
        for label in self.order:
            if label == "Positive":
                example = self.train_pos.iloc[self.index_pos]
                self.index_pos = (self.index_pos + 1) % len(self.train_pos)
                examples.append({"text": example['sentence'], "label": label})
            else:
                example = self.train_neg.iloc[self.index_neg]
                self.index_neg = (self.index_neg + 1) % len(self.train_neg)
                examples.append({"text": example['sentence'], "label": label})

        return examples

    def build_prompt(self, sentence):
        """Build prompt for a given sentence."""
        few_shot_examples = self.get_few_shot_examples()
        prompt = "<~start_set_marker~>"
        for ex in few_shot_examples:
            prompt += f"<~start_element_marker~>Review: {ex['text']}\nSentiment: {ex['label']}\n\n<~end_element_marker~>"

        prompt += f"<~end_set_marker~>Review: {sentence}\nSentiment:"
        return prompt

def load_data(test_file_path, train_file_path):
    """Load test and train datasets."""
    test_data = pd.read_csv(test_file_path, sep='\t', header=None, names=['label', 'sentence'])
    train_data = pd.read_csv(train_file_path, sep='\t', header=None, names=['label', 'sentence'])
    return test_data, train_data

def main(batch_size, model_id, order, save_file, use_set_encoding):
    # File paths
    test_file_path = "SST-2-sentiment-analysis/data/test.tsv"
    train_file_path = "SST-2-sentiment-analysis/data/train.tsv"

    # Load data into memory
    test_data, train_data = load_data(test_file_path, train_file_path)

    # Load model and tokenizer on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)
    set_encoder = SetEncoder(tokenizer)

    # Initialize dataset and dataloader
    test_dataset = SST2Dataset(test_data['sentence'], test_data['label'], train_data, order)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Get token IDs for "Positive" and "Negative"
    positive_token_id = tokenizer.encode(" Positive", add_special_tokens=False)[0]
    negative_token_id = tokenizer.encode(" Negative", add_special_tokens=False)[0]
    if positive_token_id == negative_token_id:
        # Problem with Llama2 tokenizer
        positive_token_id = tokenizer.encode(" Positive", add_special_tokens=False)[1]
        negative_token_id = tokenizer.encode(" Negative", add_special_tokens=False)[1]
    assert positive_token_id != negative_token_id

    print(f"pos_id = {positive_token_id}; neg_id = {negative_token_id}")
    print(f"sanity check: ' Positive' ->'{tokenizer.decode([positive_token_id])}'; ' Negative' -> '{tokenizer.decode([negative_token_id])}'")

    predictions = []
    y_true = []
    exactly_the_same = 0

    # Batch processing
    for prompts, label in tqdm(test_loader):

        inputs = set_encoder(prompts, device_for_output = device)

        if not use_set_encoding:
            inputs["set_pos_encoding"]  = None
            inputs["set_attention_mask"] = None
        
        with torch.no_grad():
            outputs = model(**inputs)

        # Get logits for the next token prediction
        next_token_logits = outputs.logits[:, -1, :]

        # Compare likelihoods of "Positive" and "Negative"
        positive_scores = next_token_logits[:, positive_token_id]
        negative_scores = next_token_logits[:, negative_token_id]

        batch_predictions = []
        for pos_score, neg_score in zip(positive_scores, negative_scores):
            if pos_score > neg_score:
                batch_predictions.append(1)
            elif neg_score > pos_score:
                batch_predictions.append(0)
            else:
                exactly_the_same += 1
                batch_predictions.append(random.randint(0,1))

        predictions.extend(batch_predictions)
        y_true.extend(label)

    accuracy = accuracy_score(y_true, predictions)
    f1 = f1_score(y_true, predictions)
    conf_matrix = confusion_matrix(y_true, predictions)

    # Save results
    results = {
        "model": model_id,
        "order": order,
        "accuracy": accuracy,
        "fraction_pos": sum(predictions) / len(predictions),
        "f1_score": f1,
        "confusion_matrix": conf_matrix.tolist(),
        "exactly_the_same": exactly_the_same
    }
    with open(save_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {save_file}")

def args_to_file_name(args):
    if args.use_set_encoding:
        name = "results/" + abbreviate_model_id(args.model_id) + "-set/"
    else:
        name = "results/" + abbreviate_model_id(args.model_id) + "/"
    name += abbreviate_model_id(args.model_id) + "-"
    for x in args.order:
        name += x[0] # 'p' or 'n'
    return name + ".json"

if __name__ == "__main__":

    random.seed(42)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Sentiment analysis using few-shot learning with a pre-trained model.")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-chat-hf", help='Model ID for the transformer model, one of:  "meta-llama/Llama-2-7b-chat-hf", "meta-llama/Meta-Llama-3-8B-instruct", "gpt2","mistralai/Mistral-7B-Instruct-v0.3","tiiuae/falcon-7b-instruct"')
    parser.add_argument("--order", nargs='+', default=[],
                        help="Order of examples for few-shot learning (e.g., 'Positive Negative Positive Negative').")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing data.")
    parser.add_argument('--use_set_encoding', action='store_true', help='If set_encoding should be used') # False by default

    args = parser.parse_args()

    print(f"""Running with:
    --model_id : {args.model_id}
    --order : {args.order}
    --batch_size : {args.batch_size}
    --use_set_encoding : {args.use_set_encoding}""")

    assert len([x for x in args.order if x == "Positive" or x == "Negative"]) == len(args.order)

    save_name = args_to_file_name(args)
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    main(args.batch_size, args.model_id, args.order, save_name, args.use_set_encoding)

# python main.py --model_id microsoft/Phi-3-mini-4k-instruct --order Positive Positive Negative Negative --batch_size 8 --use_set_encoding