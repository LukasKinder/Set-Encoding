from torch.utils.data import Dataset
import json
import random
from tqdm import tqdm

class HotpotDataset(Dataset):
    def __init__(self, max_tokens, tokenizer, data_path = "hotpot/hotpot_dev_distractor_v1.json"):

        self.max_tokens = max_tokens
        self.tokenizer = tokenizer

        # Load the HotpotQA dataset
        print(f"Loading data... {data_path}",end = "", flush = True)
        with open(data_path, "r") as file:
            data = json.load(file)
        print("Done")

        # 7405
        self.questions = []
        self.answers = []
        self.distracting_documents = []
        self.relevant_documents = []

        print("Processing Data...")
        for entry in tqdm(data):
            self.questions.append(entry['question'])
            a = entry['answer']
            if a == "no":
                a = "No"
            if a == "yes":
                a = "Yes"
            self.answers.append(a)
            
            supporting_facts = [x[0] for x in entry['supporting_facts']]
            self.relevant_documents.append([title + ":\n" + " ".join(sentences)  for title, sentences in entry['context'] if title in supporting_facts])
            self.distracting_documents.append([title + ":\n" + " ".join(sentences)  for title, sentences in entry['context'] if title not in supporting_facts])

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        all_documents = self.relevant_documents[idx]
        current_n_tokens = len(self.tokenizer.encode(all_documents[0]) + self.tokenizer.encode(all_documents[1]) + self.tokenizer.encode(self.questions[idx]) )
        
        n_tokens = random.randint(current_n_tokens,self.max_tokens)
        #print(n_tokens)
        #print(len(self.tokenizer.encode(self.questions[idx])))

        for i in range(0,100):
            if i < len(self.distracting_documents[idx]):
                new = self.distracting_documents[idx][i]
            else:

                random_one = self.distracting_documents[random.randint(0,len(self.questions) -1)]
                while len(random_one) == 0:
                    random_one = self.distracting_documents[random.randint(0,len(self.questions) -1)]
                new = random_one[random.randint(0, len(random_one) -1)]
            
            new_len = len(self.tokenizer.encode(new))
            print(new_len)
            if current_n_tokens + new_len > n_tokens:
                break
            current_n_tokens += new_len
            all_documents.append(new)


        random.shuffle(all_documents)
        prompt = "<~start_set_marker~>"
        for doc in all_documents:
            prompt += f"<~start_element_marker~>{doc}\n\n<~end_element_marker~>"

        prompt += f"<~end_set_marker~>\nQuestion: {self.questions[idx]}\nAnswer: {self.answers[idx]}"

        return prompt, self.answers[idx]
    
def main():
    import sys
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2",trust_remote_code=True)

    if len(sys.argv) > 1:
        dataset = HotpotDataset(4000,tokenizer,sys.argv[1])
    else:
        dataset = HotpotDataset(4000,tokenizer)
        print(len(dataset))
        input("HERE")
        dataset = HotpotDataset(2000,tokenizer, data_path= "hotpot/hotpot_train_v1.1.json")
        print(len(dataset))
        input("HERE")
    
    for i in range(100):
        print(dataset[i])
        input()

if __name__ == "__main__":
    main()