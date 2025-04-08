from litellm import completion
from typing import List

import config


def run_llm_litellm(
    llm_config: config.LLMConfig,
    messages: List[dict],
) -> str:
    """use litellm to generate json data"""
    response = completion(
        model=llm_config.model,
        messages=messages,
        api_key=llm_config.api_key,
        base_url=llm_config.base_url,
        max_tokens=llm_config.max_tokens,
        temperature=llm_config.temperature,
    )
    return response.choices[0].message.content


def singleton(cls):
    """singleton decorator"""
    instances = {}

    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return wrapper


@singleton
class HFModel:
    """HF model class"""

    def __init__(self, llm_config: config.LLMConfig):
        self.llm_config = llm_config
        self.hf_model = None
        self.init()

    def init(self):
        """init hf model"""
        from hf_utils import HfTransformer
        import torch

        self.hf_model = HfTransformer(
            task="text-generation",
            model_id=self.llm_config.hf_model,
            token=self.llm_config.hf_token,
            torch_dtype=torch.bfloat16,
        )


def run_llm_hf(
    llm_config: config.LLMConfig,
    messages: List[dict],
) -> str:
    """use hf to generate json data"""
    # if model not initialized, initialize it
    hf_model = HFModel(llm_config)
    response = hf_model.hf_model.generate(
        context=messages,
        max_new_tokens=llm_config.max_tokens,
        temperature=llm_config.temperature,
    )
    return response[0]["generated_text"][-1]["content"]


def run_llm(
    llm_config: config.LLMConfig,
    messages: List[dict],
) -> str:
    """use litellm to generate json data"""
    if llm_config.hf_model:
        return run_llm_hf(llm_config, messages)
    else:
        return run_llm_litellm(llm_config, messages)
