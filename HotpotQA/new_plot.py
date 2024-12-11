import json
import matplotlib.pyplot as plt
import numpy as np

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

def calculate_bin_accuracies(n_tokens, correct_t0, bin_size):
    max_tokens = max(n_tokens)
    bins = range(0, max_tokens + bin_size, bin_size)
    bin_accuracies = []

    for i in range(len(bins) - 1):
        bin_start = bins[i]
        bin_end = bins[i + 1]
        bin_values = [correct for token, correct in zip(n_tokens, correct_t0) if bin_start <= token < bin_end]
        if bin_values:
            avg_accuracy = sum(bin_values) / len(bin_values)
        else:
            avg_accuracy = 0
        bin_accuracies.append(avg_accuracy)

    return bins, bin_accuracies

def plot_data(json_files):
    fig, axes = plt.subplots(nrows=2, figsize=(16, 10), sharex=True)

    for idx, json_file in enumerate(json_files):
        # Load the JSON data
        with open(json_file, 'r') as file:
            data = json.load(file)
        
        # Extract fields
        n_tokens = data["n_tokens"]
        correct_t0 = data["correct_t0"]

        # Get model-specific max context size
        max_context_size = model_max_context_size(data["model"])

        # Calculate bin accuracies
        bin_size = 1024
        bins, bin_accuracies = calculate_bin_accuracies(n_tokens, correct_t0, bin_size)
        
        # Convert bins to bin centers for better visualization
        bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]

        # Bar plot for accuracy in bins
        if "set" in json_file:
            hatch_type = '//'
        else:
            hatch_type = None

        axes[idx].set_axisbelow(True)
        axes[idx].grid(True, axis="y",zorder=1)

        bars = axes[idx].bar(
            bin_centers, bin_accuracies, color="gray", width=bin_size * 0.9, edgecolor='black', hatch=hatch_type, zorder=3
        )

        # Add labels above each bar for exact accuracy values
        for bar, accuracy in zip(bars, bin_accuracies):
            height = bar.get_height()
            axes[idx].text(
                bar.get_x() + bar.get_width() / 2, height, f"{accuracy:.3f}",
                ha="center", va="bottom", fontsize=10
            )

        # Add vertical line for the max context size
        axes[idx].axvline(max_context_size, color='#ff7f00', linestyle='--',linewidth=3, label='Max Context Size')

        # Add vertical line for the max training size
        max_training_size = 4096
        axes[idx].axvline(max_training_size, color='#377eb8', linestyle='--',linewidth=3, label='Max Training size')

        if idx == 0:
            axes[idx].legend(fontsize=12)

        # Add labels and title
        axes[idx].set_ylabel('Accuracy', fontsize=14)
        if "set" in json_file:
            axes[idx].set_title("Llama3-8b with set encoding", fontsize=16)
        else:
            axes[idx].set_title("Llama3-8b without set encoding", fontsize=16)

        axes[idx].set_ylim(-0.01, 0.85)  # Fixed y-axis limits

    # Add shared x-axis label
    axes[-1].set_xlabel('Number of Tokens', fontsize=14)

    # Tight layout for spacing
    plt.tight_layout()

    # Save the combined plot
    plt.savefig("results/hoptpotQA.png")
    plt.close()
    print("Saved plot as 'results/hoptpotQA.png'")

# File paths to process
files = ["results/final/Llama3_trained-normal.json", "results/final/Llama3_trained-set.json"]

# Generate the combined plot
plot_data(files)
