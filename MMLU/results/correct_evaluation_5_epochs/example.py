import matplotlib.pyplot as plt
import numpy as np

# Data
categories = ['A', 'B', 'C', 'D']
values = [10, 15, 7, 12]

# Bar styles
colors = ['skyblue', 'lightgreen', 'salmon', 'gold']
hatches = ['/', '\\', '|', '-']

# Plotting
fig, ax = plt.subplots()
bars = []

for i, (category, value) in enumerate(zip(categories, values)):
    bar = ax.bar(
        category, 
        value, 
        color=colors[i], 
        hatch=hatches[i], 
        edgecolor='black', 
        label=f'Category {category}'
    )
    bars.append(bar)

# Adding labels and legend
ax.set_title("Bar Chart with Different Styles")
ax.set_xlabel("Categories")
ax.set_ylabel("Values")
ax.legend()

# Display the chart
plt.tight_layout()
plt.savefig("example.png")