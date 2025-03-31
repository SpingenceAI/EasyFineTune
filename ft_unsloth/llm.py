# reference from https://github.com/unslothai/unsloth/blob/main/unsloth-cli.py
import logging
import os
import yaml
from pydantic import BaseModel
from typing import Optional


from unsloth import (
    FastLanguageModel,
    to_sharegpt,
    standardize_sharegpt,
    apply_chat_template,
    is_bfloat16_supported,
)
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer


from config import (
    Config,
    ModelArgs,
    LoraArgs,
    TrainingArgs,
    ReportArgs,
    SaveArgs,
    PushArgs,
)


class DatasetArgs(BaseModel):
    dataset_name: Optional[str] = None  # Name of the dataset
    dataset_path: Optional[str] = None  # Path to the dataset
    conversation_extension: int = 1  # Number of conversation extension, default is 1.
    dataset_system_prompt: str = None  # System prompt for the dataset


class LLMConfig(Config):
    dataset: DatasetArgs = DatasetArgs()


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

fourbit_models = [
    "unsloth/mistral-7b-v0.3-bnb-4bit",  # New Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/llama-3-8b-bnb-4bit",  # Llama-3 15 trillion tokens model 2x faster!
    "unsloth/llama-3-8b-Instruct-bnb-4bit",
    "unsloth/llama-3-70b-bnb-4bit",
    "unsloth/Phi-3-mini-4k-instruct",  # Phi-3 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",  # Gemma 2.2x faster!
    "unsloth/Meta-Llama-3.1-8B",
]  # More models at https://huggingface.co/unsloth


def load_model_tokenizer(
    model_args: ModelArgs,
    lora_args: LoraArgs,
):
    """load model and tokenizer for training"""
    logger.info("Loading model and tokenizer")
    if model_args.model_name not in fourbit_models:
        raise ValueError(f"Model {model_args.model_name} is not supported")
    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_args.model_name,
        max_seq_length=model_args.max_seq_length,
        dtype=model_args.dtype,
        load_in_4bit=model_args.load_in_4bit,
    )
    # Configure PEFT model
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_args.r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.bias,
        use_gradient_checkpointing=lora_args.use_gradient_checkpointing,
        random_state=lora_args.random_state,
        use_rslora=lora_args.use_rslora,
        loftq_config=lora_args.loftq_config,
    )

    return model, tokenizer


def prepare_dataset(
    tokenizer,
    dataset_args: DatasetArgs,
):
    """prepare dataset for training"""
    logger.info("Preparing dataset")
    if dataset_args.dataset_name:
        dataset = load_dataset(dataset_args.dataset_name, split="train")
    elif dataset_args.dataset_path:
        dataset = load_dataset(
            "json", data_files=dataset_args.dataset_path, split="train"
        )
    else:
        raise ValueError("Dataset Args `dataset_name` or `dataset_path` is required")

    dataset = to_sharegpt(
        dataset,
        merged_prompt="{instruction}[[\nYour input is:\n{input}]]",
        output_column_name="output",
        conversation_extension=dataset_args.conversation_extension,  # Select more to handle longer conversations
    )
    dataset = standardize_sharegpt(dataset)

    chat_template = """Below are some instructions that describe some tasks. Write responses that appropriately complete each request.

        ### Instruction:
        {INPUT}

        ### Response:
        {OUTPUT}"""
    if dataset_args.dataset_system_prompt:
        dataset = apply_chat_template(
            dataset,
            tokenizer=tokenizer,
            chat_template=chat_template,
            default_system_message=dataset_args.dataset_system_prompt,  # "You are a helpful assistant", #<< [OPTIONAL]
        )
    else:
        dataset = apply_chat_template(
            dataset,
            tokenizer=tokenizer,
            chat_template=chat_template,
        )
    return dataset


def load_config(config_path: str) -> LLMConfig:
    """load config from yaml file"""
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return LLMConfig(**config)


def train(config_path: str):
    """train the model"""
    config = load_config(config_path)
    model_args = config.model
    lora_args = config.lora
    training_args = config.training
    report_args = config.report
    save_args = config.save
    push_args = config.push
    dataset_args = config.dataset

    logger.info("Start Training")
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_tokenizer(model_args, lora_args)
    logger.info("Preparing dataset...")
    dataset = prepare_dataset(
        tokenizer=tokenizer,
        dataset_args=dataset_args,
    )
    logger.info("Configuring training arguments...")
    # Configure training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        warmup_steps=training_args.warmup_steps,
        max_steps=training_args.max_steps,
        num_train_epochs=training_args.num_train_epochs,
        learning_rate=training_args.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=report_args.logging_steps,
        optim=training_args.optim,
        weight_decay=training_args.weight_decay,
        lr_scheduler_type=training_args.lr_scheduler_type,
        seed=training_args.seed,
        output_dir=save_args.output_dir,
        report_to=report_args.report_to,
    )
    logger.info("Training...")
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=model_args.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    # Train model
    trainer_stats = trainer.train()

    logger.info("Training stats: %s", trainer_stats)

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
