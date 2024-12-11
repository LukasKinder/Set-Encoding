import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
import sys
from scipy.stats import chisquare

def calculate_chisquare(confusion_matrix):
    
    false_a = confusion_matrix[0][1] + confusion_matrix[0][2] + confusion_matrix[0][3]
    false_b = confusion_matrix[1][0] + confusion_matrix[1][2] + confusion_matrix[1][3]
    false_c = confusion_matrix[2][0] + confusion_matrix[2][1] + confusion_matrix[2][3]
    false_d = confusion_matrix[3][0] + confusion_matrix[3][1] + confusion_matrix[3][2]

    observed  = [false_a,false_b,false_c,false_d]
    expected = [sum(observed) / 4 ] * 4 

    # Perform the chi-square goodness-of-fit test
    chi2_stat, p_value = chisquare(observed, expected)

    return p_value

def calculate_false_std(confusion_matrix):
    
    false_a = confusion_matrix[0][1] + confusion_matrix[0][2] + confusion_matrix[0][3]
    false_b = confusion_matrix[1][0] + confusion_matrix[1][2] + confusion_matrix[1][3]
    false_c = confusion_matrix[2][0] + confusion_matrix[2][1] + confusion_matrix[2][3]
    false_d = confusion_matrix[3][0] + confusion_matrix[3][1] + confusion_matrix[3][2]

    mean = (false_a + false_b + false_c + false_d) /4

    std = 0
    for f in [false_a,false_b,false_c,false_d]:
        std += (mean - f)**2
    std /= 4
    std = std**0.5 

    return std

def sort_filenames(filenames):
    filenames = sorted(filenames)

    sorted_filenames = [file for file in filenames if "finetuned" not in file]
    for i in range(0,10000,1):
        for file in filenames:
            if f"checkpoint-{i}_" in file:
                sorted_filenames.append(file)

    if len(sorted_filenames) != len(filenames):
        print(sorted_filenames)
        print(filenames)
        exit("Error while sorting")
    return sorted_filenames

# Function to load and parse .txt files from a directory
def load_experiment_data(directory):
    filenames= []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filenames.append(filename)
    
    experiments = []
    sorted_filenames = sort_filenames(filenames)
    #sorted_filenames = [filenames[1],filenames[0],filenames[2],filenames[3]]

    for filename in sorted_filenames:
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as file:
            content = file.read()
            experiments.append(json.loads(content))
    return experiments

# Function to abbreviate model IDs
def bar_name(data):
    model_id = data['model_id']
    if "llama2" in model_id.lower() or "llama-2" in model_id.lower():
        s =  "Llama2-7b"
    elif "llama3" in model_id.lower() or  "llama-3" in model_id.lower():
        s =  "Llama3-8b"
    elif "mistral" in model_id.lower():
        s =  "Mistral"
    elif "falcon" in model_id.lower():
        s =  "Falcon-7b"
    elif "phi3" in model_id.lower() or "phi-3" in model_id.lower():
        s =  "Phi-3"
    elif "pharia-1" in model_id.lower():
        s =  "Pharia-1"
    else:
        s = model_id

    if data['use_set_encoding']:
        s += " Set Encoding"
    else:
        s += " Normal Encoding"

    return s

# Function to extract relevant data based on eval_type
def extract_data(experiments, eval_type):
    data = []
    for exp in experiments:
        if exp["args"]["eval_type"] == eval_type:
            data.append({
                "model_id": exp["args"]["model_id"],
                "use_set_encoding": exp["args"]["use_set_encoding"],
                "accuracy": exp["results"]["accuracy"],
                "RTSD": exp["results"]["RTSD"],
                "confusion_matrix": exp["results"]["confusion_matrix"]
            })
    return data

def create_bar_plot(data, eval_type):
    model_ids = [f"{d['model_id'].split('-')[-1]}" if "checkpoint" in d['model_id'] else 0 for d in data]
    #model_ids = [bar_name(d)  for d in data]

    accuracies = [d["accuracy"] for d in data]
    f_stds = [calculate_false_std(d["confusion_matrix"]) for d in data]

    x = range(len(data))

    fig, ax1 = plt.subplots(figsize=(10, 6))

    bar_width = 0.4
    bars1 = ax1.bar(x, accuracies, bar_width, label='Accuracy', color='b')
    ax1.set_xlabel('Checkpoint')
    ax1.set_ylabel('Accuracy', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_ids, rotation=45, ha="right")
    ax1.set_ylim(0, 1)

    # Add accuracy labels above bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.3f}', 
                 ha='center', va='bottom', color='b', fontsize=8)

    ax2 = ax1.twinx()
    bars2 = ax2.bar([p + bar_width for p in x], f_stds, bar_width, label='RTSD', color='r')
    ax2.set_ylabel('F_STD', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_yscale('log')
    #ax2.set_ylim(0, 0.4)

    # Add RTSD labels above bars
    for idx,bar in enumerate(bars2):
        height = bar.get_height()
        p_value = calculate_chisquare(data[idx]["confusion_matrix"])
        h_position = height
        if p_value < 0.01:
            height = f"{height:.0f}**"
        elif p_value < 0.05:
            height = f"{height:.0f}*"
        else:
            height = f'{height:.0f}'

        ax2.text(bar.get_x() + bar.get_width() / 2, h_position, height, 
                 ha='center', va='bottom', color='r', fontsize=8)
    
    titel = bar_name(data[0])
    
    # titel = "Accuracy and RTSD, no finetuning"
    plt.title(titel)
    fig.tight_layout()


    print("results/" + titel)
    plt.savefig("results/" + titel)
    print(titel)


# Main script
if len(sys.argv) == 1: 
    directory = "results/"  # replace with the path to your directory
else:
    directory = sys.argv[1]

print(f"plotting data in folder: {directory}")

experiments = load_experiment_data(directory)


# Separate data based on eval_type
letter_data = extract_data(experiments, 'letter')
option_data = extract_data(experiments, 'option')

# Create bar plots for each eval_type
#create_bar_plot(letter_data, 'letter')
create_bar_plot(option_data, 'option')