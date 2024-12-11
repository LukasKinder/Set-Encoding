import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from rouge import Rouge


def calculate_rouge_scores(answers, gold_summaries):
    rouge = Rouge()
    all_rouge_scores = []
    for answer, gold_summary in zip(answers, gold_summaries):
        # Calculate ROUGE-1, ROUGE-2, and ROUGE-L
        rouge_scores = rouge.get_scores(answer, gold_summary)
        rouge1_f1 = rouge_scores[0]['rouge-1']['f']
        rouge2_f1 = rouge_scores[0]['rouge-2']['f']
        rougeL_f1 = rouge_scores[0]['rouge-l']['f']

        avg_rouge = (rouge1_f1 + rouge2_f1 + rougeL_f1) / 3

        all_rouge_scores.append(avg_rouge)

    return all_rouge_scores



def model_max_context_size(model_id):
    if "llama-2" in model_id.lower():
        return 4096
    elif "llama-3" in model_id.lower():
        return 8192
    elif "mistral" in model_id.lower():
        return 32768
    elif "falcon-7b" in model_id.lower():
        return 2048
    elif "phi-3" in model_id.lower():
        return 2048
    elif "pharia-1" in model_id.lower():
        return 8192
    elif "gpt2" in model_id.lower():
        return 1024
    else:
        return 0





def add_range(min_val, max_val, label, y_lim):
    y_loc = y_lim + 0.03
    offset = 30
    plt.vlines(x=min_val + offset, ymin=y_lim + 0.01, ymax=y_loc, lw=1, color="black", clip_on=False)
    plt.vlines(x=max_val - offset, ymin=y_lim + 0.01, ymax=y_loc, lw=1, color="black", clip_on=False)
    plt.vlines(x=min_val + (max_val - min_val) / 2, ymin=y_loc, ymax=y_loc + 0.02, lw=1, color="black", clip_on=False)
    plt.hlines(y=y_loc, xmin=min_val + offset, xmax=max_val - offset, lw=1, color="black", clip_on=False)

    plt.text(min_val + (max_val - min_val) / 2, y_loc + 0.04, label, fontsize=14, color='black', clip_on=False,
             ha='center', va='center')


def plot_data(json_file):
    # Load the JSON data from the file
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Extract the relevant fields
    n_tokens = data["n_tokens"]
    answers = data["answers"]
    gold_summaries = data["gold_summaries"]
    
    # Calculate ROUGE scores
    rouge_scores = calculate_rouge_scores(answers, gold_summaries)

    # Get model's max context size
    max_con_l = model_max_context_size(data["model"])

    y_lim = 0.5

    # Create the plot with 5:1 aspect ratio
    plt.figure(figsize=(20, 5))

    # Scatter plot with ROUGE scores
    plt.scatter(n_tokens, rouge_scores, c='blue', s=20)

    # Add labels and title
    plt.xlabel('Number of Tokens', fontsize=16)
    plt.ylabel('ROUGE Score', fontsize=16)
    
    # Title based on model and encoding type
    titel = data["model"].split("/")[-1] if "/" in data["model"] else data["model"]
    if data["set_encoding"]:
        titel += " Set Encoding"
    else:
        titel += " Normal"
    plt.title(titel, y=1.2, clip_on=False, fontsize  =20)

    # Add vertical line for the max context size
    plt.vlines(max_con_l, ymax=1, ymin=0, linestyles='dashed', label="Max Context Len")
    plt.text(max_con_l, y_lim / 2, 'max_context size', rotation=90, verticalalignment='center',
             horizontalalignment='right', fontsize=14, color='blue')
    
    avg_within = [x for x,l in zip(rouge_scores,n_tokens) if l <= max_con_l]
    avg_within = sum(avg_within) / len(avg_within)
    avg_outside = [x for x,l in zip(rouge_scores,n_tokens) if l > max_con_l]
    avg_outside = sum(avg_outside) / len(avg_outside)

    add_range(0, max_con_l, f'Avg. ROUGE: {round(avg_within,3)}', y_lim = y_lim)
    add_range(max_con_l, max(n_tokens), f'Avg. ROUGE: {round(avg_outside,3)}', y_lim = y_lim)

    # Set x-axis limit
    plt.xlim(0, max(n_tokens))
    plt.ylim(0, y_lim)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()


    # Save the plot with 5:1 aspect ratio
    filename = "results/" + "-" + os.path.basename(json_file).replace('.json', '.png')
    print(f"saved under: {filename}")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# Main logic to load and plot a single file
if __name__ == "__main__":
    json_file = sys.argv[1]
    plot_data(json_file)
