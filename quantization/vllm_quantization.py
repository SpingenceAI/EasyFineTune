from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import argparse
import yaml
from pydantic import BaseModel
from typing import Optional
import os

from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier


class QuantizationArgs(BaseModel):
    model_name: Optional[str] = None  # Name of the model
    save_path: Optional[str] = None  # Save path
    max_seq_len: int = 2048  # Max sequence length
    calibration_samples: int = 512  # Number of calibration samples
    calibration_dataset: Optional[str] = (
        "HuggingFaceH4/ultrachat_200k"  # Calibration dataset
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    return parser.parse_args()


def load_config(config_path: str) -> QuantizationArgs:
    """load config from yaml file"""
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return QuantizationArgs(**config)


def main():
    args = parse_args()
    config = load_config(args.config)
    MODEL_ID = config.model_name
    NUM_CALIBRATION_SAMPLES = config.calibration_samples
    MAX_SEQUENCE_LENGTH = config.max_seq_len
    DATASET = config.calibration_dataset
    SAVE_PATH = os.path.abspath(config.save_path)
    SAVE_PATH = os.path.join(SAVE_PATH, "vllm_quantized_model-q8_0")
    os.makedirs(SAVE_PATH, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Load and preprocess the dataset
    ds = load_dataset(DATASET, split="train_sft", streaming=False)
    ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(example["messages"], tokenize=False)
        }

    ds = ds.map(preprocess)

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize, remove_columns=ds.column_names)

    recipe = [
        SmoothQuantModifier(smoothing_strength=0.8),
        GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
    ]

    # Apply quantization
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )

    # Save the compressed model
    model.save_pretrained(SAVE_PATH, save_compressed=True)
    tokenizer.save_pretrained(SAVE_PATH)


if __name__ == "__main__":
    main()
