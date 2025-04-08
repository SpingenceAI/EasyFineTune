from transformers import pipeline
import huggingface_hub
import torch
from typing import List, Dict

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class HfTransformer:
    def __init__(self, task:str, model_id:str,token:str, torch_dtype:torch.dtype=torch.bfloat16):
        '''
        task[str]: "text-generation", "image-text-to-text", "automatic-speech-recognition", "text-classification", "summarization"
        model_id[str]
        torch_dtype[torch.dtype]: torch.bfloat16 or torch.float16
        '''
        huggingface_hub.login(token)
        self.task = task
        self.model_id = model_id
        self.torch_dtype = torch_dtype
        self.pipe = pipeline(
            self.task, model=self.model_id, model_kwargs={"torch_dtype": self.torch_dtype}, device_map="cuda"
        )

    def generate(self, context:List[Dict[str, str]], max_new_tokens:int=200, temperature=0.1):
        generate_kwargs = {
            "do_sample": True,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
        }
        if self.task == "image-text-to-text":
            return self.pipe(text=context, generate_kwargs=generate_kwargs)
        elif self.task == "text-generation":
            return self.pipe(text_inputs=context, temperature=temperature, max_new_tokens=max_new_tokens)

class Template:
    def __init__(self, context:List[Dict[str, str]]=None):
        self.context = []
    
    def add_vlm_prompt(self, role:str, text:str, image_url:str=None):
        if role not in ["system", "user", "assistant"]:
            raise ValueError(f"Invalid role: {role}")
        template = {
            "role": role,
            "content": [
                {"type": "text", "text": text}
            ]
        }
        if image_url:
            template["content"].insert(0, {"type": "image", "url": image_url})    
        self.context.append(template)

    def add_llm_prompt(self, role:str, text:str):
        if role not in ["system", "user", "assistant"]:
            raise ValueError(f"Invalid role: {role}")
        template = {"role": role, "content": text}
        self.context.append(template)
    
if __name__ == "__main__":
    llm_model_list = ["meta-llama/Llama-3.1-8B", "Qwen/Qwen2-1.5B-Instruct"]
    vlm_model_list = ["google/gemma-3-4b-it", "Qwen/Qwen2-VL-7B-Instruct"]

    msg = Template()
    msg.add_vlm_prompt("assistant", "You are a helpful assistant that can answer questions and help with tasks.")
    msg.add_vlm_prompt("user", "What animal is on the candy?", "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG")

    vlm_model = HfTransformer(task="image-text-to-text", model_id=vlm_model_list[1], torch_dtype=torch.bfloat16)
    output = vlm_model.generate(msg.context)
    print(output[0]["generated_text"][-1]["content"])

    # msg = Template()
    # msg.add_llm_prompt("assistant", "You are a helpful assistant that can answer questions and help with tasks.")
    # msg.add_llm_prompt("user", "你好")

    # llm_model = HfTransformer(task="text-generation", model_id=llm_model_list[1], torch_dtype=torch.bfloat16)
    # output = llm_model.generate(msg.context)
    # print(output)