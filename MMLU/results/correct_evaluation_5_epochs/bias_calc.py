import json
import sys

def main():

    # Read the file path from command line
    experiment_file = sys.argv[1]

    # Load the experiment data
    with open(experiment_file, 'r') as file:
        data = json.load(file)

    # Extract the confusion matrix from the data
    confusion_matrix = data.get("results", {}).get("confusion_matrix")

    total_sum = sum([sum(e) for e in confusion_matrix])


    # Print the fraction for each option
    print("Fraction of predictions for each option:")
    for i, option in enumerate(['A', 'B', 'C', 'D']):
        fraction = sum(confusion_matrix[i]) / total_sum
        print(f"Option {option}: {fraction:.2f}")

if __name__ == "__main__":
    main()
