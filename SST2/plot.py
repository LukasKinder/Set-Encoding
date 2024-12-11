import os
import json
import matplotlib.pyplot as plt
import sys
from operator import itemgetter
from matplotlib import transforms

def rainbow_text(x, y, text, colors, **kwargs):
    """
    Draws a string of text with different colors for each character.
    
    x, y : float
        The x and y coordinates for the starting position of the text.
    text : str
        The text to draw, with each character colored according to colors.
    colors : list of str
        List of colors, one for each character in text.
    **kwargs
        Additional keyword arguments passed to plt.text.
    """
    t = plt.gca().transData
    fig = plt.gcf()

    # Draw text horizontally
    for i, (char, color) in enumerate(zip(text, colors)):
        text_obj = plt.text(x + i * 0.035, y, char, color=color, transform=t, **kwargs)
        text_obj.draw(fig.canvas.get_renderer())
        ex = text_obj.get_window_extent()
        t = transforms.offset_copy(text_obj._transform, x=ex.width, units='dots')

# Get the directory from the command line argument
directory = sys.argv[1]

# Get a list of all JSON files in the directory
files = [f for f in os.listdir(directory) if f.endswith('.json')]
files.sort(reverse=True)

# Prepare to collect data for plotting
data_list = []
model_name = None

# Iterate through each file to extract data
for file in files:
    # Load the JSON data
    filepath = os.path.join(directory, file)
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Extract model name (assume all files have the same model)
    if model_name is None:
        model_name = directory.split("/")[-2]
    
    # Extract order and create colored label
    order = data["order"]
    order_label = "".join([f"P" if c == "Positive" else f"N" for c in order])
    colors = ['green' if c == 'Positive' else 'red' for c in order]
    
    # Calculate false negatives and false positives
    false_negatives = data["confusion_matrix"][1][0]
    false_positives = data["confusion_matrix"][0][1]
    
    # Calculate the difference between false negatives and false positives
    diff_fn_fp = false_positives / (false_positives + false_negatives)
    
    data_list.append((order_label, colors, diff_fn_fp, data["accuracy"]))

# Sort the data list by diff_fn_fp in descending order
#data_list.sort(key=itemgetter(2), reverse=True)

# Extract the sorted labels, colors, diff_fn_fp values, and accuracy
labels, colors_list, diff_fn_fp_values, accuracy_values = zip(*data_list)

# Create the bar plot
fig, ax = plt.subplots(figsize=(8, 5))  # Made the figure a bit smaller

# X-axis locations for the groups
x = range(len(labels))

# Plot the bars for diff_fn_fp and accuracy side by side
bar_width = 0.35
bars1 = ax.bar(x, diff_fn_fp_values, width=bar_width, label='FP / (FP + FN)', color='purple')
bars2 = ax.bar([p + bar_width for p in x], accuracy_values, width=bar_width, label='Accuracy', color='blue')

print(accuracy_values)
print(f"avg: {sum(accuracy_values) / len(accuracy_values):.2f}")
print(f"min: {min(accuracy_values):.2f}")
print(f"max: {max(accuracy_values):.2f}")
import statistics
print(f"BIAS: {statistics.stdev(diff_fn_fp_values):.2f}")

# Add labels and title
ax.set_title(model_name)

# Set x-ticks
ax.set_xticks([p + bar_width / 2 for p in x])
ax.set_xticklabels([''] * len(labels))  # Set empty labels to avoid overlap

# Add colored labels using the rainbow_text function
for i in range(len(labels)):
    rainbow_text(x[i] + bar_width / 2 - 0.2, -0.05, labels[i], colors_list[i], size=12, ha='center')

# Add value labels on top of each bar
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom', fontsize=10, color='black')

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom', fontsize=10, color='black')

# Set y-axis limits
ax.set_ylim(min(min(diff_fn_fp_values), 0), max(max(diff_fn_fp_values), 1))

# Add a legend
ax.legend()

# Add vertical grid lines for easier comparison
ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.7)

# Save the figure
plt.tight_layout()
print(f"save under: " + "results/" + model_name + ".png")
plt.savefig("results/" + model_name + ".png")
