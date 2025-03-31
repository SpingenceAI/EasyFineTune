# reference from https://github.com/unslothai/unsloth/blob/main/unsloth-cli.py
# reference from https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb
import logging
import os
from typing import Optional
import json


from pydantic import BaseModel
import yaml
from PIL import Image as PILImage
from datasets import Dataset, Image
from unsloth import (
    is_bfloat16_supported,
    FastVisionModel,
)

from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, load_from_disk


from unsloth.trainer import UnslothVisionDataCollator

from config import (
    Config,
    ModelArgs,
    LoraArgs,
    TrainingArgs,
    ReportArgs,
    SaveArgs,
    PushArgs,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DatasetArgs(BaseModel):
    # custom dataset
    dataset_path: Optional[str] = (
        None  # jsonl file path, each line is a json object, each json object has "image" and "text" fields
    )
    dataset_image_dir: Optional[str] = None  # image directory
    # huggingface dataset
    dataset_name: Optional[str] = None
    dataset_system_prompt: Optional[str] = "Describe the image in detail."

    def get_dataset_image_dir(self):
        if self.dataset_path:
            if self.dataset_image_dir:
                return self.dataset_image_dir
            else:
                return os.path.dirname(self.dataset_path)
        return None


class VisionModelArgs(BaseModel):
    model_name: str = "unsloth/Qwen2-VL-7B-Instruct"
    load_in_4bit: bool = True
    lora_r: int = 16
    random_seed: int = 3407
    hf_token: str = None


class VisionLoraArgs(BaseModel):
    finetune_vision_layers: bool = True
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True


class VLMConfig(Config):
    dataset: DatasetArgs = DatasetArgs()
    vision_lora: VisionLoraArgs = VisionLoraArgs()


def load_config(config_path: str) -> VLMConfig:
    """load vlm config from yaml file"""
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return VLMConfig(**config)


SUPPORTED_MODELS = [
    "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",  # Llama 3.2 vision support
    "unsloth/Llama-3.2-11B-Vision-bnb-4bit",
    "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit",  # Can fit in a 80GB card!
    "unsloth/Llama-3.2-90B-Vision-bnb-4bit",
    "unsloth/Pixtral-12B-2409-bnb-4bit",  # Pixtral fits in 16GB!
    "unsloth/Pixtral-12B-Base-2409-bnb-4bit",  # Pixtral base model
    "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",  # Qwen2 VL support
    "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
    "unsloth/Qwen2-VL-72B-Instruct-bnb-4bit",
    "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit",  # Any Llava variant works!
    "unsloth/llava-1.5-7b-hf-bnb-4bit",
    "unsloth/Qwen2-VL-7B-Instruct",
]  # More models at https://huggingface.co/unsloth


def load_model_tokenizer(
    model_args: ModelArgs, lora_args: LoraArgs, vision_lora_args: VisionLoraArgs
):
    """load model and tokenizer for training"""
    if model_args.model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Model {model_args.model_name} is not supported")
    # Load model and tokenizer
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_args.model_name,
        load_in_4bit=model_args.load_in_4bit,  # Use 4bit to reduce memory use. False for 16bit LoRA.
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
    )
    # Configure PEFT model
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=vision_lora_args.finetune_vision_layers,  # False if not finetuning vision layers
        finetune_language_layers=vision_lora_args.finetune_language_layers,  # False if not finetuning language layers
        finetune_attention_modules=vision_lora_args.finetune_attention_modules,  # False if not finetuning attention layers
        finetune_mlp_modules=vision_lora_args.finetune_mlp_modules,  # False if not finetuning MLP layers
        r=lora_args.r,  # The larger, the higher the accuracy, but might overfit
        lora_alpha=lora_args.lora_alpha,  # Recommended alpha == r at least
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.bias,
        random_state=lora_args.random_state,
        use_rslora=lora_args.use_rslora,  # We support rank stabilized LoRA
        loftq_config=lora_args.loftq_config,  # And LoftQ
        # target_modules = "all-linear", # Optional now! Can specify a list if needed
    )
    return model, tokenizer


def convert_dataset(dataset_path: str, dataset_image_dir: str, save_path: str):
    """convert dataset to huggingface dataset
    dataset_path: jsonl file path, each line is a json object, each json object has "image" and "text" fields
    dataset_image_dir: image directory
    save_path: save path
    """
    logger.info(f"converting dataset from {dataset_path} to {save_path}")
    dataset_dict = {"image": [], "text": [], "image_path": []}
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            image_path = os.path.join(dataset_image_dir, item["file_name"])
            if os.path.exists(image_path):
                try:
                    # make sure the image is a valid image
                    with PILImage.open(image_path) as img:
                        dataset_dict["image"].append(image_path)
                        dataset_dict["text"].append(item["text"])
                        dataset_dict["image_path"].append(image_path)
                except Exception as e:
                    logger.error(f"error {image_path}: {e}")
                    continue

    dataset = Dataset.from_dict(dataset_dict)
    dataset = dataset.cast_column("image", Image())
    dataset.save_to_disk(save_path)


def load_custom_dataset(dataset_args: DatasetArgs):
    """load custom dataset"""
    if dataset_args.dataset_path:
        dataset = load_dataset(
            "json", data_files=dataset_args.dataset_path, split="train"
        )
    else:
        raise ValueError("dataset_path is required")
    return dataset


def convert_to_conversation(sample: dict, dataset_system_prompt: str):
    """convert the sample to a conversation"""
    instruction = sample.get("instruction", dataset_system_prompt)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": sample["image"]},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": sample["text"]}]},
    ]
    return {"messages": conversation}


def prepare_dataset(dataset_args: DatasetArgs):
    """prepare dataset for training"""

    if dataset_args.dataset_path:
        if not os.path.exists(dataset_args.dataset_path):
            raise FileNotFoundError(f"dataset {dataset_args.dataset_path} not found")
        dataset_image_dir = dataset_args.get_dataset_image_dir()
        if dataset_image_dir is None:
            raise ValueError("dataset_image_dir is required")
        if not os.path.exists(dataset_image_dir):
            raise FileNotFoundError(f"image directory {dataset_image_dir} not found")
        logger.info(f"Start to load dataset from {dataset_args.dataset_path}")
        save_path = os.path.join(
            os.path.dirname(dataset_args.dataset_path), "converted_dataset"
        )
        os.makedirs(save_path, exist_ok=True)
        convert_dataset(dataset_args.dataset_path, dataset_image_dir, save_path)
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"converted dataset {save_path} not found")
        logger.info(f"Start to load converted dataset from {save_path}")
        dataset = load_from_disk(save_path)
    else:
        dataset = load_dataset(dataset_args.dataset_name, split="train")

    # convert dataset to conversation format for training
    converted_dataset = [
        convert_to_conversation(sample, dataset_args.dataset_system_prompt)
        for sample in dataset
    ]

    return converted_dataset


def train(config_path: str):
    """train the model for vlm using unsloth"""
    config = load_config(config_path)
    model_args = config.model
    lora_args = config.lora
    vision_lora_args = config.vision_lora
    training_args = config.training
    report_args = config.report
    save_args = config.save
    push_args = config.push

    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_tokenizer(model_args, lora_args, vision_lora_args)
    FastVisionModel.for_training(model)  # Enable for training!

    logger.info("Preparing dataset...")
    dataset = prepare_dataset(config.dataset)

    logger.info("Configuring training arguments...")
    # configure training arguments
    args = SFTConfig(
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        warmup_steps=training_args.warmup_steps,
        # max_steps = 30,
        num_train_epochs=training_args.num_train_epochs,  # Set this instead of max_steps for full training runs
        learning_rate=float(training_args.learning_rate),
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=report_args.logging_steps,
        optim=training_args.optim,
        weight_decay=training_args.weight_decay,
        lr_scheduler_type=training_args.lr_scheduler_type,
        seed=training_args.seed,
        output_dir=save_args.output_dir,
        report_to="none",  # For Weights and Biases
        # You MUST put the below items for vision finetuning:
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_num_proc=4,
        max_seq_length=model_args.max_seq_length,
    )
    # load SFT trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),  # Must use!
        train_dataset=dataset,
        args=args,
    )

    trainer_stats = trainer.train()

    logger.info(f"Training stats: {trainer_stats}")

    # Save model
    if save_args.save_model:
        # if args.quantization_method is a list, we will save the model for each quantization method
        if save_args.save_gguf:
            if isinstance(save_args.quantization, list):
                for quantization_method in save_args.quantization:
                    logger.info(
                        f"Saving model with quantization method: {quantization_method}"
                    )
                    model.save_pretrained_gguf(
                        save_args.save_path,
                        tokenizer,
                        quantization_method=quantization_method,
                    )
                    if push_args.push_model:
                        model.push_to_hub_gguf(
                            hub_path=push_args.hub_path,
                            hub_token=push_args.hub_token,
                            quantization_method=quantization_method,
                        )
            else:
                logger.info(
                    f"Saving model with quantization method: {save_args.quantization}"
                )
                model.save_pretrained_gguf(
                    save_args.save_path,
                    tokenizer,
                    quantization_method=save_args.quantization,
                )
                if push_args.push_model:
                    model.push_to_hub_gguf(
                        hub_path=push_args.hub_path,
                        hub_token=push_args.hub_token,
                        quantization_method=save_args.quantization,
                    )
        else:
            model.save_pretrained_merged(
                save_args.save_path, tokenizer, save_args.save_method
            )
            if push_args.push_model:
                model.push_to_hub_merged(
                    save_args.save_path, tokenizer, push_args.hub_token
                )
    else:
        logger.warning("Warning: The model is not saved!")
