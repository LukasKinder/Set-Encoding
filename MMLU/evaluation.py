# Import libraries
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import DataLoader

from set_encoding_utils import SetEncoder
from mmlu_dataloader import MMLUDataset
from prompt_utils import QuestionConstructor
import random


class MMLU_Evaluator():
    
    def __init__(self, model_id,data_dir,eval_type,use_set_encoding,batch_size,max_batches,tokenizer_path,first_tokens_differ,model = None):

        self.eval_type = eval_type
        self.use_set_encoding = use_set_encoding
        self.batch_size = batch_size
        self.max_batches = max_batches
        self.first_tokens_differ = first_tokens_differ

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.set_encoder = SetEncoder(self.tokenizer)

        if "pharia" in model_id.lower():
            self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,trust_remote_code=True,device_map = "auto")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,device_map = "auto")
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        print("Done")

        q_c = QuestionConstructor(model_id,use_set_markers= True,use_letters=eval_type == "letter",include_begin_of_text_tokens= False)

        # Create the dataset with filtering enabled
        dataset = MMLUDataset(data_dir,q_c.construct_question)
        if eval_type == 'option' and not first_tokens_differ:
            # one 4rth of th batch size because every question is run 4 times with one of the options as the answer
            self.dataloader = DataLoader(dataset, batch_size= int(batch_size / 4), shuffle=True)
        else:
            self.dataloader = DataLoader(dataset, batch_size= batch_size, shuffle=True)
        print(f"number of questions = {len(dataset)}")

    def calculate_RTSD(self,confusion_matrix):
        recalls = []
        for i in range(4):
            recalls.append(confusion_matrix[i][i] / sum([confusion_matrix[j][i] for j in range(4)]))
        average_recall = sum(recalls) / len(recalls)
        return ( sum([(r - average_recall)**2 for r in recalls]) / 4)**0.5
    
    def evaluation_letter(self):
        totoal_correct = 0
        n_questions = 0
        confusion_mattix = [[0,0,0,0] for i in range(4)]
        map_letter_index = {'A' : 0, 'B' : 1, 'C':2, 'D' : 3}
        n_invalid = 0
        invalid_answers = []

        for n_batch, batch in enumerate(self.dataloader):
            print(f"Batch {n_batch} of {len(self.dataloader)}", end = "... ",flush= True)
            if n_batch == self.max_batches:
                break

            questions = batch['question']
            correct_answers = batch['answer']

            tokens = self.set_encoder(questions, self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **tokens,
                    max_new_tokens=1,
                    pad_token_id = 666
                )

            # Decode the generated tokens
            answers = [self.tokenizer.decode(output,skip_special_tokens=True) for output in outputs]
            answer_letters = [answer[-1] for answer in answers]

            n_correct = 0
            for answer, correct_option in zip(answer_letters,correct_answers):
                if answer == correct_option:
                    n_correct +=1
                try:
                    confusion_mattix[map_letter_index[answer]][map_letter_index[correct_option]] +=1
                except KeyError:
                    invalid_answers.append(answer)
                    print(f"Key error: {answer}")
                    n_invalid +=1
                

            totoal_correct += n_correct
            n_questions += len(answers)
            print(f"performance: {n_correct} / {self.batch_size}")

        rtsd = self.calculate_RTSD(confusion_mattix)

        return {"n_correct" : totoal_correct, "n_questions" : n_questions, "RTSD" : rtsd, "accuracy" : totoal_correct / n_questions,
                "confusion_matrix" : confusion_mattix, "n_invalid" : n_invalid, "invalid_answers" : invalid_answers}

    def evaluation_option_differernt_tokens(self):
        totoal_correct = 0
        n_questions = 0
        confusion_mattix = [[0,0,0,0] for i in range(4)]
        map_letter_index = {'A' : 0, 'B' : 1, 'C':2, 'D' : 3}
        result = []

        for n_batch, batch in enumerate(self.dataloader):
            print(f"Batch {n_batch} of {len(self.dataloader)}", end = "... ",flush= True)
            if n_batch == self.max_batches:
                break

            questions = batch['question']

            tokens = self.set_encoder(questions, self.model.device)
            if not self.use_set_encoding:
                tokens["set_attention_mask"] = None
                tokens["set_pos_encoding"] = None

            len_tokens = tokens["input_ids"].shape[1]
            print(f"len tokens = {len_tokens} ...", end = "... ",flush= True)

            with torch.no_grad():
                output = self.model(**tokens)

            n_correct = 0
            for i in range(len(questions)):

                #print(questions[i])

                first_tokens = []
                for j in range(4):
                    tokens_option = self.tokenizer.tokenize(" " + batch['options'][j][i])
                    while tokens_option[0] == '▁':
                        tokens_option = tokens_option[1:] # Because the Llama2 tokenizer is wierd
                    first_tokens.append(tokens_option[0])
                
                #print(first_tokens)
                first_tokens = [int(self.tokenizer.convert_tokens_to_ids(t)) for t in first_tokens]
                
                if len(list(set(first_tokens))) != 4:
                    print("No unique tokens!!!")
                    for j in range(4):
                        print(" " + batch['options'][j][i])
                        print(self.tokenizer.tokenize(" " + batch['options'][j][i]))
                    print(first_tokens)
                    print("Error, first tokens are not the same")
                    continue
                

                max_logprob = -99999999999999
                max_letters = []
                for l,first_token in zip(["A","B","C","D"],first_tokens):
                    logprob = output.logits[i][len_tokens-1][first_token]
                    #print(logprob)
                    if logprob > max_logprob:
                        max_logprob = logprob
                        max_letters = [l]
                    elif logprob == max_logprob:
                        max_letters.append(l)
                answer = random.choice(max_letters)

                correct_option = batch['answer'][i]

                if answer == correct_option:
                    n_correct +=1
                    result.append(1)
                    #print("correct")
                else:
                    result.append(0)
                    #print("false")
                #input("----")
                n_questions += 1
                confusion_mattix[map_letter_index[answer]][map_letter_index[correct_option]] +=1

                

            totoal_correct += n_correct
            print(f"performance: {n_correct} / {self.batch_size}")

        rtsd = self.calculate_RTSD(confusion_mattix)

        return {"n_correct" : totoal_correct, "n_questions" : n_questions, "RTSD" : rtsd, "accuracy" : totoal_correct / n_questions,
                "confusion_matrix" : confusion_mattix, "questions_correct" : result}

    def evaluation_option(self):

        totoal_correct = 0
        n_questions = 0
        confusion_mattix = [[0,0,0,0] for i in range(4)]
        map_letter_index = {'A' : 0, 'B' : 1, 'C':2, 'D' : 3}

        for n_batch, batch in enumerate(self.dataloader):
            print(f"Batch {n_batch} of {len(self.dataloader)}", end = "... ",flush= True)

            if n_batch == self.max_batches:
                break

            questions = []
            for i,question in  enumerate(batch['question']):
                for j in range(4):
                    questions.append(question + batch['options'][j][i] )
            correct_answers = batch['answer']

            tokens = self.set_encoder(questions, self.model.device)
            with torch.no_grad():
                outputs = self.model(
                    **tokens
                )
            
            log_probs = torch.nn.functional.log_softmax(outputs["logits"], dim=-1)

            log_probs_for_tokens = []
            for i in range(len(tokens["input_ids"])):
                sum_probs = 0
                for idx,token_id in enumerate(tokens["input_ids"][i]):
                    if token_id == 128009 or token_id == 128000 or idx == 0:
                        continue
                    sum_probs += log_probs[i][idx-1,token_id]
                log_probs_for_tokens.append(sum_probs)

            n_correct = 0
            for i in range(len(correct_answers)):
                max = -99999999999999
                best = "Z"
                for j, l in enumerate(["A","B","C","D"]):
                    if log_probs_for_tokens[i*4 + j] > max:
                        max = log_probs_for_tokens[i*4 + j]
                        best = l
                if best == correct_answers[i]:
                    n_correct +=1
                confusion_mattix[map_letter_index[best]][map_letter_index[correct_answers[i]]] +=1
            
            totoal_correct +=n_correct
            n_questions += len(correct_answers)

            print(f"performance: {n_correct} / {self.batch_size / 4}")
        

        rtsd = self.calculate_RTSD(confusion_mattix)
        acc = totoal_correct / n_questions

        return {"n_correct" : totoal_correct, "n_questions" : n_questions, "RTSD" : rtsd, "accuracy" : acc,
                "confusion_matrix" : confusion_mattix}

    def evaluation_free(self):
        raise NotImplementedError
    
    def __call__(self):
        if self.eval_type == "letter":
            results = self.evaluation_letter()
        elif self.eval_type == "option" and not self.first_tokens_differ:
            results = self.evaluation_option()
        elif self.eval_type == "option" and self.first_tokens_differ:
            results = self.evaluation_option_differernt_tokens()
        else:
            results = self.evaluation_free()
        return results
        
