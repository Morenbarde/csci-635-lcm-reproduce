from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

model_name = "mistralai/Mistral-Small-24B-Instruct-2501"


class Democritus_Model():
    
    def __init__(self):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, fix_mistral_regex=True)

        self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                dtype=torch.float16,
            )

    
    def generate_response(self, prompt: str, report_time: bool = False) -> str:

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        start_time = time.time()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256, 
            temperature=0,
            do_sample=False
        )

        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        if report_time:
            print("Response Time: ", time.time() - start_time)

        return response