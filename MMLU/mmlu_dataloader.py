import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from prompt_utils import PromptParams

class MMLUDataset(Dataset):
    def __init__(self, data_dir, construct_question_function, proportion=None,filter_unshufflable=False,max_tokens=None,tokenizer = None
                 ,max_len_type = "question"):
        self.data_dir = data_dir
        self.data = []
        self.construct_question_function = construct_question_function

        self.filter_unshufflable = filter_unshufflable
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        assert max_tokens == None or tokenizer != None
        self.max_len_type = max_len_type
        assert max_len_type in ["question","letter","correct_option","all_options"]

        self.proportion = proportion if proportion else [0.25, 0.25, 0.25, 0.25]
        if sum(self.proportion) != 1:
            print(f"WARNING: Proportions for A, B, C, D should add up to 1, but sum of {self.proportion} is {sum(self.proportion)}")
        self._load_data()

    def max_len_prompt(self, row):

        letter_index_map = {"A" : 0, "B" : 1, "C" : 2, "D" : 3}

        if self.max_len_type == "question":
            prompt =  self.construct_question_function(row[0],row[1:5])
        elif self.max_len_type == "letter":
            prompt =  self.construct_question_function(row[0],row[1:5]) + "A"
        elif self.max_len_type == "correct_option":
            prompt =  self.construct_question_function(row[0],row[1:5]) + row[1 + letter_index_map[row[5]]]
        elif self.max_len_type == "all_options":
            max_l = 0
            for option in row[1:5]:
                prompt =  self.construct_question_function(row[0],row[1:5]) + " " + option
                tokens_option = self.tokenizer.tokenize(prompt)
                if len(tokens_option) > max_l:
                    max_l = len(tokens_option)
            return max_l
        else:
            print("Error: invalid max_len type!")
            exit()

        tokens = self.tokenizer.tokenize(prompt)

        if '<~start_set_marker~>' in tokens:
            return len(tokens) - 10
        else:
            return len(tokens)


    def _load_data(self):
        unshufflable_strings = PromptParams().unshufflable_strings

        for file_name in os.listdir(self.data_dir):
            n_too_long = 0
            n_total = 0
            n_unshufflable = 0
            if file_name.endswith('.csv'):
                print(f"{file_name}: ", end = "", flush = True)
                file_path = os.path.join(self.data_dir, file_name)
                df = pd.read_csv(file_path, header=None)
                n_q = df.values.tolist()
                for row in n_q:
                    if any(type(option) != str for option in row[1:5]):
                        row[1:5] = [str(option) for option in row[1:5]]

                    if self.filter_unshufflable and any(any(s.lower() in option.lower() for s in unshufflable_strings) for option in row[1:5]):
                        n_unshufflable +=1
                        continue

                    if self.max_tokens != None:
                        if len(row[0].split(" ")) > self.max_tokens:
                            #The question already has too many words, dont even bother to tokenize this
                            n_too_long +=1
                            continue
                        
                        if self.max_len_prompt(row) > self.max_tokens:
                            n_too_long +=1
                            continue
                    n_total +=1
                    self.data.append(row)
            print(f"Of the {len(n_q)} questions {n_total} are used. ( too long = {n_too_long}, unshufflable = {n_unshufflable})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        question = row[0]
        options = row[1:5]
        correct_answer = row[5]

        #find the correct option
        correct_option = options[ord(correct_answer) - ord('A')]

        # Determine the new correct option based on the specified proportions
        new_correct_index = np.random.choice([0, 1, 2, 3], p=self.proportion)

        # Shuffle the options
        random.shuffle(options)

        # Swap the correct option to be at the new_correct_index
        options[options.index(correct_option)] = options[new_correct_index]
        options[new_correct_index] = correct_option

        return {
            'question' : self.construct_question_function(question,options),
            'answer' : ['A', 'B', 'C', 'D'][new_correct_index],
            'options' : options
        }