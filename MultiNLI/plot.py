import os
import json
import matplotlib.pyplot as plt
import sys
from operator import itemgetter
from matplotlib import transforms

from scipy.spatial.distance import cosine
import numpy as np

def average_pairwise_similarity(vectors):
    num_vectors = len(vectors)
    total_similarity = 0
    count = 0

    for i in range(num_vectors):
        for j in range(i + 1, num_vectors):
            # Calculate cosine similarity
            similarity = cosine(vectors[i], vectors[j])  # 1 - cosine distance to get similarity
            total_similarity += similarity
            count += 1

    # Average of all pairwise similarities
    return total_similarity / count

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
if directory[-1] != '/':
    directory += '/'

# Get a list of all JSON files in the directory
files = [f for f in os.listdir(directory) if f.endswith('.json')]

# Prepare to collect data for plotting
data_list = [None for i in range(6)]
order_map = {"ENC" : 0,"ECN" : 1,"NEC" : 2,"NCE" : 3,"CEN" : 4,"CNE" : 5}
model_name = None

# Iterate through each file to extract data
vectors = []
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
    order_label = "".join([c[0].upper() for c in order])
    order_color_mapping = {"entailment" : "green", "neutral" : "yellow", "contradiction" : "red"}
    colors = [order_color_mapping[c] for c in order]

    c_m = data["confusion_matrix"]
    n_total = sum([sum(e) for e in c_m])
    n_entailment = ( c_m[0][0] + c_m[1][0] + c_m[2][0] ) / n_total
    n_neutral = ( c_m[0][1] + c_m[1][1] + c_m[2][1] ) / n_total
    n_contradiction = ( c_m[0][2] + c_m[1][2] + c_m[2][2] ) / n_total
    vectors.append([n_entailment,n_neutral,n_contradiction])
    
    data_list[order_map[order_label]] = (order_label, colors, data["accuracy"],n_entailment,n_neutral,n_contradiction)

print(f"Average pairwise distance = {average_pairwise_similarity(vectors):.3f}")

# Sort the data list by accuracy in descending order
#data_list.sort(key=itemgetter(2), reverse=True)

# Extract the sorted labels, colors, accuracy values, and false classification values
labels, colors_list, accuracy_values, n_entailments, n_neutrals, n_contradictions = zip(*data_list)

print(f"Average accuracy = {sum(accuracy_values) / 6:.2f}")
print(f"Min accuracy = {min(accuracy_values):.2f}")
print(f"Max accuracy = {max(accuracy_values):.2f}")

# Create the bar plot
fig, ax = plt.subplots(figsize=(12, 5))  # Made the figure a bit smaller

# X-axis locations for the groups
x = range(0, len(labels) * 2, 2)

# Plot the accuracy bars
bar_width = 0.6
bars1 = ax.bar(x, accuracy_values, width=bar_width, label='Accuracy', color='blue')

# Plot the stacked bars for false_entailments, false_neutrals, and false_contradictions
bars_fe = ax.bar([p + bar_width for p in x], n_entailments, width=bar_width, label='e', color='green', bottom=0)
bars_fn = ax.bar([p + bar_width for p in x], n_neutrals, width=bar_width, label='n', color='yellow', bottom=n_entailments)
bars_fc = ax.bar([p + bar_width for p in x], n_contradictions, width=bar_width, label='c', color='red', bottom=[fe + fn for fe, fn in zip(n_entailments, n_neutrals)])

# Add value labels on top of each stacked segment
for i in range(len(bars_fe)):
    # Label for false_entailment
    height_fe = bars_fe[i].get_height()
    ax.text(bars_fe[i].get_x() + bars_fe[i].get_width() / 2, height_fe / 2, f'{height_fe:.2f}', ha='center', va='center', fontsize=12, color='black')
    
    # Label for false_neutral
    height_fn = bars_fn[i].get_height()
    ax.text(bars_fn[i].get_x() + bars_fn[i].get_width() / 2, height_fe + height_fn / 2, f'{height_fn:.2f}', ha='center', va='center', fontsize=12, color='black')
    
    # Label for false_contradiction
    height_fc = bars_fc[i].get_height()
    ax.text(bars_fc[i].get_x() + bars_fc[i].get_width() / 2, height_fe + height_fn + height_fc / 2, f'{height_fc:.2f}', ha='center', va='center', fontsize=12, color='black')


# Add labels and title
ax.set_title(f"MultiNLI - {model_name}")

# Set x-ticks
ax.set_xticks([p + bar_width / 2 for p in x])
ax.set_xticklabels([''] * len(labels))  # Set empty labels to avoid overlap

# Add colored labels using the rainbow_text function
for i in range(len(labels)):
    rainbow_text(x[i], -0.05, labels[i], colors_list[i], size=14, ha='center')

# Add value labels on top of each bar
for bars in [bars1]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom', fontsize=12, color='black')

# Set y-axis limits
ax.set_ylim(0, 1)

# Add a legend
ax.legend(fontsize = 12)

# Add vertical grid lines for easier comparison
ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.7)

plt.ylabel("", fontsize=14)  # Increase font size here

# Save the figure
plt.tight_layout()
print(f"save under: " + "results/" + model_name + ".png")
plt.savefig("results/" + model_name + ".png")
