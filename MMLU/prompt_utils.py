
class PromptParams:
    unshufflable_strings = [
        "A and B","A and C","B and C","A or B","A or C","B or C", "neither A nor B", 
        "None of the above", "Either of these", "Neither of these", "None of these", "all of the above",
        "All options","both ", "All of", "all the above","All of these"
    ]

def special_tokens(model_id,token_type):
    assert token_type in ["start","end"]

    if "lama-2" in model_id.lower() or "lama2" in model_id.lower():
        if token_type == 'start':
            return "[INST]"
        return "[/INST]"
    
    if "lama-3" in model_id.lower() or "pharia-1" in model_id.lower() or "lama3" in model_id.lower():
        if token_type == 'start':
            return "<|start_header_id|>user<|end_header_id|>\n\n"
        return "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    if "gpt2" in model_id.lower():
        if token_type == 'start':
            return ""
        return "\n"
    
    if "mistral" in model_id.lower():
        if token_type == 'start':
            return "[INST]"
        return "[/INST]"
    
    if "falcon" in model_id.lower():
        if token_type == 'start':
            return ">>QUESTION<<"
        return ">>ANSWER<<"
    
    if "phi-3" in model_id.lower() or "phi3" in model_id.lower():
        if token_type == 'start':
            return "<|user|>\n"
        return "<|end|>\n<|assistant|>"
    
    print("Unknown model regarding special tokens!!!")
    print(f" Model is: '{model_id.lower()}'")
    raise KeyError

def begin_of_text_tokens(model_id):
    if "lama-2" in model_id.lower() or "gpt2" in model_id.lower() or "mistral-7b" in model_id.lower():
        return "<s>"

    
    if "lama-3" in model_id.lower():
        return "<|begin_of_text|>"
    
    
    if "falcon-7b" in model_id.lower():
        return "<|endoftext|>"
    
    if "mistral" in model_id.lower():
        return "<s"
    
    print("Unknown model regarding special tokens!")
    print(f" Model is: '{model_id.lower()}'")
    raise KeyError



class QuestionConstructor():

    def __init__(self,model_id, use_set_markers, use_letters =False,include_begin_of_text_tokens = True):
        self.model_id = model_id
        self.use_set_markers = use_set_markers
        self.include_begin_of_text_tokens = include_begin_of_text_tokens
        self.use_letters = use_letters


        if use_set_markers:
            self.set_markers = {
                "beginning" : "<~start_set_marker~><~start_element_marker~>",
                "seperator" : "<~end_element_marker~><~start_element_marker~>",
                "end" : "<~end_element_marker~><~end_set_marker~>"
            }
        else:
            self.set_markers = {"beginning" : "","seperator" : "","end" : ""}

    def construct_question(self,question, options):
        if self.use_letters:
            question = f"""{special_tokens(self.model_id,"start")}{question}
(A) {options[0]}
(B) {options[1]}
(C) {options[2]}
(D) {options[3]}
{special_tokens(self.model_id,"end")}Answer: ("""
        else:
            question = f"""{special_tokens(self.model_id,"start")}{question}
{self.set_markers["beginning"]} * {options[0]}
{self.set_markers["seperator"]} * {options[1]}
{self.set_markers["seperator"]} * {options[2]}
{self.set_markers["seperator"]} * {options[3]}
{self.set_markers["end"]}{special_tokens(self.model_id,"end")}The correct answer is:"""

        if self.include_begin_of_text_tokens:
            question = begin_of_text_tokens(self.model_id) + question
        return question
