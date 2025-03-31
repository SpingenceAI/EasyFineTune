from typing import Optional
from pydantic import BaseModel


# configs for generate data
class LLMConfig(BaseModel):
    """LLM config for litellm"""

    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: Optional[int] = 4096
    temperature: Optional[float] = 0.5


class GenerateConfig(BaseModel):
    """Generate config for generate questions"""

    generate_questions_batch_num: Optional[int] = 3


class Args(BaseModel):
    """Args config for generate data"""

    chunk_size: Optional[int] = 1024
    chunk_overlap: Optional[int] = 200
    min_chunk_size: Optional[int] = 512
    generate_questions_batch_num: Optional[int] = 3
    generate_num: Optional[int] = 2


class Config(BaseModel):
    """Config for generate data"""

    llm: LLMConfig
    args: Args

    def __str__(self):
        return self.model_dump_json(indent=4)
