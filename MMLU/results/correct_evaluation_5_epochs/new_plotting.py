import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

all_data = {
    "Llama2-7b" : {
        "normal" : {
            "No Finetuning":    {"bias" : 0.0, "accuracy" : 0.4038317},
            "No Bias" :         {"bias" : 1.5408845918435664e-16, "accuracy" : 0.60587210},
            "Low Bias" :        {"bias" : 2.8073576527245707e-75, "accuracy" : 0.591938293107},
            "Medium Bias":      {"bias" : 0.0, "accuracy" : 0.56332421000},
            "High Bias":        {"bias" : 0.0, "accuracy" : 0.263}
        }, 
        "set" : {
            "No Finetuning":    {"bias" : 0.17430012, "accuracy" : 0.355063},
            "No Bias" :         {"bias" : 0.4062566, "accuracy" : 0.57924856},
            "Low Bias" :        {"bias" : 0.36200824, "accuracy" : 0.576511570},
            "Medium Bias":      {"bias" : 0.5264487, "accuracy" : 0.5809902},
            "High Bias":        {"bias" : 0.893445897, "accuracy" : 0.579497387409}
        }, 
    },
    "Llama3-8b" : {
        "normal" : {
            "No Finetuning":    {"bias" : 0.004352, "accuracy" : 0.7536700},
            "No Bias" :         {"bias" : 0.614383, "accuracy" : 0.7643692},
            "Low Bias" :        {"bias" : 1.4358084377003993e-08, "accuracy" : 0.7606369743},
            "Medium Bias":      {"bias" : 1.4183484376161845e-44, "accuracy" : 0.747947250},
            "High Bias":        {"bias" : 0.0, "accuracy" : 0.263}
        }, 
        "set" : {
            "No Finetuning":    {"bias" : 0.844220307, "accuracy" : 0.68424981},
            "No Bias" :         {"bias" : 0.95238353, "accuracy" : 0.742224},
            "Low Bias" :        {"bias" : 0.5150502, "accuracy" : .74321970},
            "Medium Bias":      {"bias" : 0.85559203, "accuracy" : 0.741975615824},
            "High Bias":        {"bias" : 0.89910718, "accuracy" : 0.744961433}
        } 
    },
    "Falcon-7b"  : {
        "normal" : {
            "No Finetuning":    {"bias" : 0.0, "accuracy" : 0.2848967},
            "No Bias" :         {"bias" : 1.3532440647653916e-05, "accuracy" : 0.3968648},
            "Low Bias" :        {"bias" : 3.5970223120599976e-129, "accuracy" : 0.3834287136103},
            "Medium Bias":      {"bias" : 0.0, "accuracy" : 0.34287136},
            "High Bias":        {"bias" : 0.0, "accuracy" : 0.2540432943518288}
        }, 
        "set" : {
            "No Finetuning":    {"bias" : 0.232724, "accuracy" : 0.28415028614},
            "No Bias" :         {"bias" : 0.664053721, "accuracy" : 0.3988554366},
            "Low Bias" :        {"bias" : 0.326413053557904, "accuracy" : 0.39860661},
            "Medium Bias":      {"bias" : 0.2362623, "accuracy" : 0.3981089},
            "High Bias":        {"bias" : 0.7402234, "accuracy" : 0.39910425}
        } 
    },
    "Mistral-7b" : {
        "normal" : {
            "No Finetuning":    {"bias" : 2.345565776447454e-06, "accuracy" : .69843244588},
            "No Bias" :         {"bias" : 0.001314325, "accuracy" : 0.7414779},
            "Low Bias" :        {"bias" : 0.0065039, "accuracy" : 0.73028116446},
            "Medium Bias":      {"bias" : 9.838822060054556e-59, "accuracy" : 0.7275441},
            "High Bias":        {"bias" : 0.0, "accuracy" : 0.238616571}
        }, 
        "set" : {
            "No Finetuning":    {"bias" : 0.51104886701651, "accuracy" : 0.605374471},
            "No Bias" :         {"bias" : 0.773000074, "accuracy" : 0.73177407},
            "Low Bias" :        {"bias" : 0.78564618750, "accuracy" : 0.730032346},
            "Medium Bias":      {"bias" : 0.4512349764, "accuracy" : 0.722567802},
            "High Bias":        {"bias" : 0.6841843, "accuracy" : 0.7297835}
        } 
    }
}

training_conditions = ["No Finetuning","No Bias","Low Bias","Medium Bias","High Bias"]
models = ["Llama2-7b","Llama3-8b","Mistral-7b","Falcon-7b"]
conditions = ["normal", "set"]


# Create the subplots with wider figures
fig, axs = plt.subplots(2, 2, figsize=(14, 8))  # Increase figure width
axs = axs.flatten()


# Generate bar plots for each subplot
for figure_n, (ax, model) in enumerate(zip(axs, all_data.keys())):
    bar_width = 0.4  # Width of bars for each condition
    x = np.arange(len(training_conditions))  # Positions for training conditions
    
    for i, condition in enumerate(conditions):
        accuracies = []
        biases = []
        
        for bias_label in training_conditions:
            accuracies.append(all_data[model][condition][bias_label]["accuracy"])
            biases.append(all_data[model][condition][bias_label]["bias"])
        

        bars = []
        for j,(acc_instance, p_value_instance) in enumerate(zip(accuracies,biases)):

            if p_value_instance > 0.05:
                c = '#ff7f00'
            else:
                c = '#377eb8'

            if condition == "normal":
                hatch_type = None
            else:
                hatch_type = '//'

            bar = ax.bar(
                j + i * bar_width,
                acc_instance,
                width=bar_width,
                color=c, 
                hatch=hatch_type, 
                edgecolor='black'
            )
            bars.append(bar)
        
        
        # Add labels above the bars
        for bar_container, acc in zip(bars, accuracies):
            for rect in bar_container.patches:  # Iterate over individual bars in the container
                label = f"{acc:.3f}"  # Format the label
                ax.text(
                    rect.get_x() + rect.get_width() / 2,  # Center of the individual bar
                    rect.get_height() + 0.01,  # Slightly above the bar
                    label,
                    ha='center',
                    va='bottom',
                    fontsize=10
                )

    
    ax.set_title(model)
    ax.set_xticks(x + bar_width / 2)  # Center ticks between grouped bars
    if figure_n > 1:
        ax.set_xticklabels(training_conditions, rotation=45,fontsize = 12)
    else:
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
    if figure_n % 2 == 0:
        ax.set_ylabel("Accuracy",fontsize = 12)
    else:
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    if figure_n == 0:
        # Create legend elements for colors
        color_legend_elements = [
            Patch(facecolor='#377eb8', edgecolor='black', label='Biased'),  # Red bar for "Set"
            Patch(facecolor='#ff7f00', edgecolor='black', label='Unbiased')  # Green bar for "Normal"
        ]

        hashed_legend_elements = [
            Patch(facecolor='white', edgecolor='black', hatch='///', label='Set'),  # Red bar for "Set"
            Patch(facecolor='white', edgecolor='black', label='Normal')  # Green bar for "Normal"
        ]

        # Combine the legends + hashed_legend_elements

        legend_elements = color_legend_elements + hashed_legend_elements

        # Add legend to the plot
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    ax.set_axisbelow(True)
    ax.grid(True, axis="y")
    ax.set_ylim(0, 0.9)  # Fixed y-axis limits

# Add legend for color meaning
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#4CAF50", label="p > 0.05 (Green)"),
    Patch(facecolor="#FFC107", label="0.01 ≤ p ≤ 0.05 (Yellow)"),
    Patch(facecolor="#F44336", label="p < 0.01 (Red)")
]
fig.legend(handles=legend_elements, loc="upper left", fontsize=10, bbox_to_anchor=(0.95, 0.01))

# Adjust layout
plt.tight_layout()  # Leave space for legend
#plt.suptitle("Accuracy by Training Condition and P-value", fontsize=16)
plt.savefig("mmlu_all_results_simplified.png")

if True:
        
    colors = ["blue", "green"]
    markers = ["o", "s", "^", "D", "P"]

    training_conditions = ["No Finetuning","No Bias","Low Bias","Medium Bias","High Bias"]
    models = ["Llama2-7b","Llama3-8b","Mistral-7b","Falcon-7b"]
    conditions = ["normal", "set"]

    # Create the subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    # Generate scatter plots for each subplot
    for ax,model in zip(axs,models):
        
        for i,condition in enumerate(conditions):

            for j,bias in enumerate(training_conditions):
                
                acc = all_data[model][condition][bias]["accuracy"]
                bias = all_data[model][condition][bias]["bias"] 
                
                ax.scatter(
                    [bias], 
                    [acc], 
                    color=colors[i], 
                    marker=markers[j], 
                    s=100
                )

        ax.set_title(model)
        ax.set_xlabel("Bias")
        ax.set_ylabel("Accuracy")
        ax.grid(True)
        ax.set_xlim(0, 1)
        #ax.set_ylim(0, 1)
        #ax.set_xscale("log")

    # Add shared legend
    fig.legend(conditions, loc="upper center", ncol=5, fontsize=10)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    #plt.suptitle("Scatter Plots: Bias vs. Accuracy", fontsize=16)
    plt.savefig("mmlu_all_results.png")
