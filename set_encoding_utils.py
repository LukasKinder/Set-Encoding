import torch
from transformers import SpecialTokensMixin

special_tokens_map = {
    "meta-llama/Llama-2-7b-chat-hf" : {"start" : "[INST]", "end" : "[/INST]"},
    "meta-llama/Meta-Llama-3-8B-instruct" : {"start" : "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant.<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n", 
                                             "end" : "<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"},
    "gpt2" : {"start" : "", "end" : "\n\n"},
    "mistralai/Mistral-7B-Instruct-v0.3" : {"start" : "[INST]",
                                             "end" : "[/INST]"},
    "tiiuae/falcon-7b-instruct" : {"start" : ">>QUESTION<<",
                                   "end" : ">>ANSWER<<"},
    "microsoft/Phi-3-mini-4k-instruct" : {"start" : "<|user|>\n",
                                          "end" : "<|end|>\n<|assistant|>"},                   
}

class SetEncoder():

    def __init__(self, tokenizer,custom_size = None):
        self.tokenizer = tokenizer

        pad_token = self.tokenizer.eos_token
        if pad_token == None:
            self.pad_token_id = 0 # does not matter anyways
        else:
            try:
                self.pad_token_id = self.tokenizer([pad_token])['input_ids'][0][1]
            except IndexError:
                self.pad_token_id = self.tokenizer([pad_token])['input_ids'][0][0]

        markers = SpecialTokensMixin()._set_markers
        try:
            marker_ids = [x[1]  for x in self.tokenizer(markers)['input_ids']]
        except IndexError:
            marker_ids = [x[0]  for x in self.tokenizer(markers)['input_ids']]

        self.set_markers = {}
        for marker, id in zip(markers,marker_ids):
            self.set_markers[marker] = id
        self.custom_size = custom_size


    def encode_sequence(self,tokens,size,pos_encodings,set_attention_mask):
        total_size = len(pos_encodings)

        # the beginning tokens that are unused because prompt may not be longest in batch
        unused_size = int(total_size - (size))

        # add padding
        tokens = [ self.pad_token_id for i in range(unused_size)] + tokens
        # padded tokens can attend to themself to preven NaN values
        set_attention_mask[0:unused_size,0:unused_size] = True

        #staring point is a traditional triangular attention mask used for causal attention
        mask = torch.full((size, size), 1.0)
        mask = torch.tril(mask, diagonal=0)
        set_attention_mask[unused_size:,unused_size:] = mask.bool()
        
        current_index = unused_size
        current_pos = 0
        
        while self.set_markers["<~start_set_marker~>"] in tokens:
        
            # pre-set tokens can attend everything
            size_pre_set = tokens.index(self.set_markers["<~start_set_marker~>"]) - current_index
            pos_encodings[current_index:current_index + size_pre_set] = torch.arange(start = current_pos, end = current_pos + size_pre_set)
            current_pos += size_pre_set
            current_index += size_pre_set

            assert tokens[current_index] == self.set_markers["<~start_set_marker~>"]
            tokens.pop(current_index)

            max_size = 0
            total_size = 0
            index_set_starts = current_index

            while tokens[current_index] != self.set_markers["<~end_set_marker~>"]:
                assert tokens[current_index] == self.set_markers["<~start_element_marker~>"]
                tokens.pop(current_index)
                size_element = tokens.index(self.set_markers["<~end_element_marker~>"]) - current_index
                # pos is reset for every element in the set
                pos_encodings[current_index:current_index + size_element] = torch.arange(start = current_pos, end = current_pos + size_element)
                #mask attention such that the element can not attend other elements
                set_attention_mask[current_index:current_index + size_element,index_set_starts:current_index] = False
                
                # Note the maximum position within the set
                current_index +=size_element
                if size_element > max_size:
                    max_size = size_element

                assert tokens[current_index] == self.set_markers["<~end_element_marker~>"]
                tokens.pop(current_index)

            #after the set the position is the lowest available position
            current_pos += max_size

            assert tokens[current_index] == self.set_markers["<~end_set_marker~>"]
            tokens.pop(current_index)
        
        for marker_id in self.set_markers.values():
            assert marker_id not in tokens # There should not be any markers left

        # There are no more sets in the prompt, the last tokens can attend everything
        size_rest = len(pos_encodings[current_index:])
        pos_encodings[current_index:current_index + size_rest] = torch.arange(start = current_pos, end = current_pos + size_rest)

        return torch.tensor(tokens,dtype = torch.int64)



    def __call__(self, prompts, device_for_output = "cpu"):

        if type(prompts) == str:
            prompts = [prompts]

        all_tokens = []
        prompt_sizes  = []

        for prompt in prompts:
            # work with lists, easier but unefficient :(
            tokens = [int(t) for t in self.tokenizer(prompt, return_tensors='pt')['input_ids'][0]]
            all_tokens.append(tokens)
            # The actual size does not include the set-markers
            prompt_sizes.append(len([t for t in tokens if t not in self.set_markers.values()]))

        max_size = max(prompt_sizes)
        if self.custom_size != None:
            if max_size > self.custom_size:
                print(f"Error, actual length is {max_size}, but it should not be longer than {self.custom_size}")
                exit()
            max_size = self.custom_size

        batch_size = len(prompts) # The batch size

        pos_encodings = torch.ones(batch_size,max_size, dtype = torch.int64) # The position of each okens, resets for each element of a set
        attention_mask = torch.ones(batch_size,max_size) # the attention mask that blocks attention to padding tokens (only relevant for batches)
        set_attention_masks = torch.zeros(batch_size,max_size,max_size).bool() # the attention mask that blocks attention between elements of a set

        new_tokens = torch.empty(batch_size,max_size,dtype = torch.int64) 

        for i in range(batch_size):
            attention_mask[i,:max_size - prompt_sizes[i]] = 0 # No attention to padded tokens
            new_tokens[i] = self.encode_sequence(all_tokens[i],prompt_sizes[i],pos_encodings[i],set_attention_masks[i])

        return {"input_ids" : new_tokens.to(device_for_output), "set_attention_mask" : set_attention_masks.to(device_for_output), 
                "set_pos_encoding" : pos_encodings.to(device_for_output), "attention_mask" : attention_mask.to(device_for_output)}
        