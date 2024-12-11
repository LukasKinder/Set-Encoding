import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def model_max_context_size(model_id):
    if "llama-2" in model_id.lower():
        return 4096
    elif "llama-3" in model_id.lower():
        return 8192
    elif "mistral" in model_id.lower():
        return 32768
    elif "falcon-7b" in model_id.lower():
        return 2048
    elif "phi-3" in model_id.lower() or "phi3" in model_id.lower():
        return 2048
    elif "pharia-1" in model_id.lower():
        return 8192
    elif "gpt2" in model_id.lower():
        return 1024
    else:
        return 8192
    
def accuracy_short_long(n_tokens, correct_t0, max_context_size):
    within = []
    outside = []
    for n_token, correct in zip(n_tokens, correct_t0):
        if n_token > max_context_size:
            outside.append(correct)
        else:
            within.append(correct)
    
    acc_within = sum(within) / len(within) if within else 0
    acc_outside = sum(outside) / len(outside) if outside else 0
    return acc_within, acc_outside

def add_range(min,max,label):
    y_loc = 1.03
    offset = 30
    plt.vlines(x = min+offset, ymin = 1.01, ymax = y_loc, lw=1, color = "black",clip_on=False)
    plt.vlines(x = max-offset, ymin = 1.01, ymax = y_loc, lw=1, color = "black",clip_on=False)
    plt.vlines(x = min + (max-min) /2, ymin = y_loc, ymax = y_loc + 0.02, lw=1, color = "black",clip_on=False)
    plt.hlines(y = y_loc, xmin= min+offset, xmax=max-offset, lw=1, color = "black",clip_on=False)

    plt.text(min + (max-min) / 2, y_loc+ 0.05, label,fontsize=16, color='black',clip_on=False,ha='center', va='center')
    


def plot_data(json_file):
    # Load the JSON data from the file
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    # Extract the relevant fields
    n_tokens = data["n_tokens"]
    logprobs = data["logprobs"]
    correct_t0 = data["correct_t0"]

    max_con_l = model_max_context_size(data["model"])
    acc_within, acc_outside = accuracy_short_long(n_tokens, correct_t0, max_con_l)
    
    # Convert logprobs to a numpy array for better handling
    logprobs = np.array(logprobs)
    
    # Create the plot with 5:1 aspect ratio
    plt.figure(figsize=(20, 5))
    
    # Scatter plot with conditional colors based on correct_t0 values
    colors = ['green' if correct else 'red' for correct in correct_t0]
    plt.scatter(n_tokens, logprobs, c=colors, s = 30)
    
    # Add labels and title
    plt.xlabel('Number of Tokens', fontsize=18)
    plt.ylabel('Probability', fontsize=18)

    x = [2000, 4000, 6000 ,8000,10000, 12000 ,14000, 16000]
    plt.xticks(x, [f"{val//1000}k" for val in x],fontsize=16)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    try:
        titel = f'{data["model"].split("/")[1]}'
    except IndexError:
        titel = f'{data["model"].split("/")[0]}'
    if "-set" in json_file:
        titel += " Set Encoding"
    else:
        titel += " Normal"
    #plt.title(titel, y = 1.1,clip_on=False, fontsize=18)

    # Add vertical line for the max context size
    plt.vlines(max_con_l, ymax=1, ymin=0, linestyles='dashed', label="Max Context Len")
    plt.text(max_con_l, 0.5, 'max_context size', rotation=90, verticalalignment='center', 
             horizontalalignment='right', fontsize=16, color='blue')
    
    add_range(0,max_con_l,f'Acc: {acc_within:.2f}')
    add_range(max_con_l,max(n_tokens), f'Acc: {acc_outside:.2f}')

    # Set x-axis limit
    plt.xlim(0, max(n_tokens))
    plt.ylim(0, 1)

    # Save the plot with 5:1 aspect ratio
    filename = "results/" + "-" + json_file.split("/")[-1][:-5] 
    filename += ".png"
    print(f"saved under: {filename}")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Main logic to iterate through the files
path = sys.argv[1]
dir = os.listdir(path)

for root, d_names, f_names in os.walk(path):
    for f_name in f_names:
        f_path = root + "/" + f_name
        if ".json" in f_name:
            print(f_path)
            plot_data(f_path)
