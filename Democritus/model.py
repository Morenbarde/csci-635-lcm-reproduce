from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
import torch
import time

model_name = "mistralai/Mistral-Small-24B-Instruct-2501"


class Democritus_Model():
    
    def __init__(self):

        # quantization_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        # )

        from packaging import version
        if version.parse(transformers.__version__) < version.parse("4.48.0"):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, fix_mistral_regex=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name, fix_mistral_regex=True)

        self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                dtype=torch.float16,
                # quantization_config=quantization_config
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