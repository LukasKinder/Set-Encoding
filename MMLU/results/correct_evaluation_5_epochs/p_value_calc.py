import json
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



def read_data(path):
    with open(path, 'r') as file:
        content = file.read()
        data = json.loads(content)
    return data
    

# Main script
path = sys.argv[1]

print(f"plotting data in file: {path}")

data = read_data(path)

acc = data["results"]["accuracy"]
p_value = calculate_chisquare( data["results"]["confusion_matrix"] )

print(f"Accuracy: { acc }")
print(f"p-value: {p_value}")