# Experiment with SST2 Dataset

This project runs experiments on the SST2 dataset. The order of examples for few-shot learning can be adapted. When the `--use_set_encoding` flag is set, there will be no bias based on the order of examples.

## Running the Experiment

To run the experiment, execute for example:

python main.py --model_id microsoft/Phi-3-mini-4k-instruct --order Positive Positive Negative Negative --batch_size 8 --use_set_encoding

This will run the experiment with microsoft/Phi-3-mini-4k-instruct with set encoding and prompt containing four examples, ordered as follows:

Positive
Positive
Negative
Negative

Results from the experiment will be saved in a folder named results/model_Name/, where model_Name corresponds to the model you are using.

# Plotting the results

To visualize the results, you can generate a bar plot by running the following command:

python plot <path_to_folder_with_results>

This will create a bar plot that compares the results from different experiments, providing a clear visual representation of the performance.