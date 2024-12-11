from torch.utils.data import Dataset
import random
from tqdm import tqdm


SPECIAL_TOKENS_MAP = {
    "meta-llama/Llama-2-7b-chat-hf" : {"start" : "[INST]", "end" : "[/INST]"},
    "meta-llama/Meta-Llama-3-8B-instruct" : {"start" : "<|start_header_id|>user<|end_header_id|>\n\n", 
                                             "end" : "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"},
    "gpt2" : {"start" : "", "end" : "\n\n"},
    "mistralai/Mistral-7B-Instruct-v0.3" : {"start" : "[INST]",
                                             "end" : "[/INST]"},
    "tiiuae/falcon-7b-instruct" : {"start" : ">>QUESTION<<",
                                   "end" : ">>ANSWER<<"},
    "microsoft/Phi-3-mini-4k-instruct" : {"start" : "<|user|>\n",
                                          "end" : "<|end|>\n<|assistant|>"},
    "Aleph-Alpha/Pharia-1-LLM-7B-control-hf" : {"start" : "<|start_header_id|>user<|end_header_id|>\n\n", 
                                                "end" : "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"}                      
}

def n_tokens_documents(documents, tokenizer):
    n = 0
    max = 0
    all_sizes = []
    for document in documents:
        n_this = len(tokenizer.encode(document))
        all_sizes.append(n_this)
        n += n_this
        if n_this > max:
            max = n_this
    return n,max,all_sizes

class MultiNewsDataset(Dataset):
    def __init__(self, bucket_size, max_total_tokens, max_tokens_document, tokenizer, special_tokens_map, data_path = "file_exchange/test.txt" , bucket_range = 1000):

        self.max_tokens_total = max_total_tokens
        self.tokenizer = tokenizer
        self.bucket_size = bucket_size
        self.max_tokens_document = max_tokens_document
        self.special_tokens_map = special_tokens_map
        assert "start" in special_tokens_map.keys() and "end" in special_tokens_map.keys()

        with open(data_path + ".src","r") as f:
            all_articles = f.readlines()
        all_articles = [articles.split("story_separator_special_tag") for articles in all_articles]

        with open(data_path + ".tgt","r") as f:
            summaries = f.readlines()

        #shuffle data
        temp = list(zip(all_articles, summaries))
        random.shuffle(temp)
        all_articles, summaries = zip(*temp)
        all_articles, summaries = list(all_articles), list(summaries)

        buckets = [0 for _ in range(int(max_total_tokens / bucket_range))]
        sizes = [[] for _ in range(int(max_total_tokens / bucket_range))]

        self.filtered_articles = []
        self.filtered_summaries = []
        all_sizes = []
        max_size_documents = 0

        print("Processing Data...")
        for articles,summary in tqdm(zip(all_articles,summaries),total=len(summaries)):
            n_tokens, _, sizes_documents = n_tokens_documents(articles,self.tokenizer)
            if n_tokens > self.max_tokens_total:
                continue

            idx_bucket = int(n_tokens / bucket_range)
            if buckets[idx_bucket] >= bucket_size:
                continue

            buckets[idx_bucket] +=1
            sizes[idx_bucket].append((n_tokens,sizes_documents))

            self.filtered_articles.append(articles)
            self.filtered_summaries.append(summary)
            all_sizes.append(n_tokens)
            if n_tokens > max_size_documents:
                max_size_documents = n_tokens

        for i in range(int(max_total_tokens / bucket_range)):
            #print(f"Len Range {bucket_range * i} - {bucket_range*(i+1)}: {buckets[i]}, -> {[ (x[0],len(x[1])) for x in sizes[i]]}")
            #print(f"Len Range {bucket_range * i} - {bucket_range*(i+1)}: {buckets[i]}, -> {sizes[i]}")
            print(f"Len Range {bucket_range * i} - {bucket_range*(i+1)}: {buckets[i]},")
        print(f"len dataset = {len(self.filtered_articles)}")

        for x in self.filtered_articles:
            random.shuffle(x)

        # move the largest document to the beginning, this way, potential out-of-memory errors will appear in the beginning
        idx_largest = all_sizes.index(max_size_documents)
        self.filtered_articles.insert(0,self.filtered_articles[idx_largest])
        self.filtered_summaries.insert(0,self.filtered_summaries[idx_largest])
        self.filtered_articles.pop(idx_largest +1)
        self.filtered_summaries.pop(idx_largest +1)


    def __len__(self):
        return len(self.filtered_summaries)

    def __getitem__(self, idx):
        
        articles = self.filtered_articles[idx]
        summary = self.filtered_summaries[idx]

        prompt = self.special_tokens_map["start"] + "<~start_set_marker~>"
        for i,article in enumerate(articles):
            total_tokens = 0
            prompt += "<~start_element_marker~>"
            article_sentences = article.split(".")
            print(f"Artikle {i}:")
            for sentence in article_sentences:
                n_tokens_sentence = len(self.tokenizer.encode(sentence)) + 2
                if total_tokens + n_tokens_sentence > self.max_tokens_document:
                    prompt += "<~end_element_marker~><~start_element_marker~>"
                    total_tokens = 0
                    print("split",end = " ")

                prompt += sentence + "."
                total_tokens += n_tokens_sentence
            prompt += "\n\n<~end_element_marker~>"


        prompt += "<~end_set_marker~>"
        
        prompt += "\n\nWrite a ~300 word summary about the given articles." + self.special_tokens_map["end"]

        return prompt, summary
    
def main():
    from transformers import AutoTokenizer

    model_id = "meta-llama/Meta-Llama-3-8B-instruct"

    
    tokenizer = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)

    test_dataset = MultiNewsDataset(bucket_size= 10,max_total_tokens=100000,max_tokens_document=100000,tokenizer=tokenizer,special_tokens_map=SPECIAL_TOKENS_MAP[model_id])
    input("Confirm")

    for i in range(100):
        articles, summary = test_dataset[i]
        #print(articles,summary)
        input()

if __name__ == "__main__":
    main()