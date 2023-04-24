import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class Generator:
    def __init__(self, model_path, fp16, gpu) -> None:
        self.gpu = gpu
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        if fp16:
            self.model = self.model.half()
        if gpu:
            self.model = self.model.to(device)
    
    def generate(self, input, max_length, return_sequences, beams, temperature) -> str:
        encoded_input = self.tokenizer(input, return_tensors="pt")
        if self.gpu:
            input_ids, attention_mask = encoded_input['input_ids'].to(device), encoded_input['attention_mask'].to(device)
        else:
            input_ids, attention_mask = encoded_input['input_ids'], encoded_input['attention_mask']
        result = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length, num_return_sequences=return_sequences, num_beams=beams, temperature=temperature)
        output = ""
        for res in result:
            output = output + self.tokenizer.decode(res) + "\n"
        return output
