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

# MultiNLI Dataset Class
class MultiNLIDataset(Dataset):
    def __init__(self, premises, hypotheses, labels, train_data, order, contextfree_hypothesis  ="N/A"):
        self.premises = premises
        self.hypotheses = hypotheses
        self.labels = labels
        self.train_ent = train_data[train_data['gold_label'] == 'entailment']
        self.train_neu = train_data[train_data['gold_label'] == 'neutral']
        self.train_con = train_data[train_data['gold_label'] == 'contradiction']
        self.index_ent = 0
        self.index_neu = 0
        self.index_con = 0
        self.order = order
        self.contextfree_hypothesis = contextfree_hypothesis

    def __len__(self):
        return len(self.premises)

    def __getitem__(self, idx):
        premise = self.premises[idx]
        hypothesis = self.hypotheses[idx]
        label = self.labels[idx]
        prompt,no_context_prompt = self.build_prompt(premise, hypothesis)
        return prompt, no_context_prompt, label

    def get_few_shot_examples(self):
        """Get random few-shot examples based on the specified order."""
        examples = []
        for label in self.order:
            if label == "entailment":
                example = self.train_ent.iloc[self.index_ent]
                self.index_ent = (self.index_ent + 1) % len(self.train_ent)
            elif label == "neutral":
                example = self.train_neu.iloc[self.index_neu]
                self.index_neu = (self.index_neu + 1) % len(self.train_neu)
            else:  # contradiction
                example = self.train_con.iloc[self.index_con]
                self.index_con = (self.index_con + 1) % len(self.train_con)
            examples.append({"premise": example['sentence1'], "hypothesis": example['sentence2'], "label": label})
        return examples

    def build_prompt(self, premise, hypothesis):
        """Build prompt for a given premise and hypothesis."""
        few_shot_examples = self.get_few_shot_examples()
        prompt = "<~start_set_marker~>"
        for ex in few_shot_examples:
            prompt += f"<~start_element_marker~>Premise: {ex['premise']}\nHypothesis: {ex['hypothesis']}\nLabel: {ex['label']}\n\n<~end_element_marker~>"

        normal_prompt = prompt + f"<~end_set_marker~>Premise: {premise}\nHypothesis: {hypothesis}\nLabel:"
        context_free_prompt = prompt + f"<~end_set_marker~>Premise: {premise}\nHypothesis: {self.contextfree_hypothesis}\nLabel:"

        return normal_prompt, context_free_prompt

def load_data(test_file_path, train_file_path):
    """Load test and train datasets."""
    test_data = pd.read_json(test_file_path, lines=True)
    train_data = pd.read_json(train_file_path, lines=True)
    return test_data, train_data

def args_to_file_name(args):
    
    name = "results/" + abbreviate_model_id(args.model_id)
    if args.callibrate:
        name += "-callibrated/"

    if args.use_set_encoding:
        name += "-set/"
    else:
        name += "/"
    name += abbreviate_model_id(args.model_id) + "-"
    for x in args.order:
        name += x[0] # 'e', 'n', or 'c'
    return name + ".json"

def callibrate(model,inputs, id_e,id_c,id_n):
    with torch.no_grad():
            outputs = model(**inputs)

    # Get logits for the next token prediction
    next_token_logits = outputs.logits[:, -1, :]

    # Compare likelihoods of "Entailment", "Neutral", and "Contradiction"
    entailment_scores = next_token_logits[:, id_e]
    neutral_scores = next_token_logits[:, id_n]
    contradiction_scores = next_token_logits[:, id_c]

    for i,(ent_score, neu_score, con_score) in enumerate(zip(entailment_scores, neutral_scores, contradiction_scores)):
        max_score = max(ent_score, neu_score, con_score).clone().detach()
        entailment_scores[i] = max_score -ent_score
        neutral_scores[i] = max_score - neu_score
        contradiction_scores[i] = max_score - con_score

    return entailment_scores, neutral_scores, contradiction_scores




def main(batch_size, model_id, order, save_file, use_set_encoding):
    # File paths
    test_file_path = "multinli_1.0/multinli_1.0_dev_matched.jsonl"
    train_file_path = "multinli_1.0/multinli_1.0_train.jsonl"

    # Load data into memory
    test_data, train_data = load_data(test_file_path, train_file_path)

    # Load model and tokenizer on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "pharia" in model_id.lower():
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,trust_remote_code=True).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)
    set_encoder = SetEncoder(tokenizer)

    # Initialize dataset and dataloader
    test_dataset = MultiNLIDataset(test_data['sentence1'], test_data['sentence2'], test_data['gold_label'], train_data, order)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Get token IDs for "Entailment", "Neutral", and "Contradiction"
    entailment_token_id = tokenizer.encode(" entailment", add_special_tokens=False)[0]
    neutral_token_id = tokenizer.encode(" neutral", add_special_tokens=False)[0]
    contradiction_token_id = tokenizer.encode(" contradiction", add_special_tokens=False)[0]
    if len(set([entailment_token_id, neutral_token_id, contradiction_token_id])) != 3:
        # problem with llama2 tokenizer:
        entailment_token_id = tokenizer.encode(" entailment", add_special_tokens=False)[1]
        neutral_token_id = tokenizer.encode(" neutral", add_special_tokens=False)[1]
        contradiction_token_id = tokenizer.encode(" contradiction", add_special_tokens=False)[1]

    print(f"entailment_id = {entailment_token_id}; neutral_id = {neutral_token_id}; contradiction_id = {contradiction_token_id}")
    print(f"sanity check: ' entailment' ->'{tokenizer.decode([entailment_token_id])}'; ' neutral' -> '{tokenizer.decode([neutral_token_id])}'; ' contradiction' -> '{tokenizer.decode([contradiction_token_id])}'")

     # Check for token collisions
    assert len(set([entailment_token_id, neutral_token_id, contradiction_token_id])) == 3

    predictions = []
    y_true = []
    exactly_the_same = 0

    # Batch processing
    for prompts, no_context_prompts ,label in tqdm(test_loader):

        inputs = set_encoder(prompts, device_for_output = device)

        if not use_set_encoding:
            inputs["set_pos_encoding"]  = None
            inputs["set_attention_mask"] = None
        
        with torch.no_grad():
            outputs = model(**inputs)

        # Get logits for the next token prediction
        next_token_logits = outputs.logits[:, -1, :]

        # Compare likelihoods of "Entailment", "Neutral", and "Contradiction"
        entailment_scores = next_token_logits[:, entailment_token_id]
        neutral_scores = next_token_logits[:, neutral_token_id]
        contradiction_scores = next_token_logits[:, contradiction_token_id]

        if args.callibrate:
            batch_bias_e,batch_bias_n,batch_bias_c = callibrate(model,set_encoder(no_context_prompts, device_for_output = device),entailment_token_id,neutral_token_id,contradiction_token_id)
            entailment_scores += batch_bias_e
            neutral_scores += batch_bias_n
            contradiction_scores += batch_bias_c

        batch_predictions = []
        for ent_score, neu_score, con_score in zip(entailment_scores, neutral_scores, contradiction_scores):

            max_score = max(ent_score, neu_score, con_score)
            options_with_max_score = [label for score,label in zip([ent_score, neu_score, con_score],['entailment', 'neutral', 'contradiction']) if score == max_score]
            if len(options_with_max_score) != 1:
                exactly_the_same +=1
            random.shuffle(options_with_max_score)
            batch_predictions.append(options_with_max_score[0])


        predictions.extend(batch_predictions)
        y_true.extend(label)

    accuracy = accuracy_score(y_true, predictions)
    f1 = f1_score(y_true, predictions, average='weighted')
    conf_matrix = confusion_matrix(y_true, predictions, labels=['entailment', 'neutral', 'contradiction'])

    # Save results
    results = {
        "model": model_id,
        "order": order,
        "accuracy": accuracy,
        "fraction_entailment": predictions.count('entailment') / len(predictions),
        "fraction_neutral": predictions.count('neutral') / len(predictions),
        "fraction_contradiction": predictions.count('contradiction') / len(predictions),
        "f1_score": f1,
        "confusion_matrix": conf_matrix.tolist(),
        "exactly_the_same": exactly_the_same
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
    parser.add_argument("--order", nargs="+", default=["entailment", "neutral", "contradiction"], help="Order of few-shot examples: ['entailment', 'neutral', 'contradiction']. You can experiment with different orders.")
    parser.add_argument("--use_set_encoding", action="store_true", help="Use set encoding.")
    parser.add_argument("--callibrate", action="store_true", help="If callibration should be used.")
    args = parser.parse_args()

    assert all(item in {"entailment", "neutral", "contradiction"} for item in args.order)

    save_file = args_to_file_name(args)
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    main(args.batch_size, args.model_id, args.order, save_file, args.use_set_encoding)

## "meta-llama/Llama-2-7b-chat-hf", "meta-llama/Meta-Llama-3-8B-instruct", "gpt2","mistralai/Mistral-7B-Instruct-v0.3","tiiuae/falcon-7b-instruct", "microsoft/Phi-3-mini-4k-instruct", "Aleph-Alpha/Pharia-1-LLM-7B-control-hf"

# python main.py --model_id Aleph-Alpha/Pharia-1-LLM-7B-control-hf --order entailment neutral contradiction --batch_size 8 --use_set_encoding