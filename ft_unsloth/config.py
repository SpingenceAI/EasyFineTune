# reference from https://github.com/unslothai/unsloth/blob/main/unsloth-cli.py
from pydantic import BaseModel
from typing import Optional


class ModelArgs(BaseModel):
    """
    model_group = parser.add_argument_group("🤖 Model Options")
    model_group.add_argument('--model_name', type=str, default="unsloth/llama-3-8b", help="Model name to load")
    model_group.add_argument('--max_seq_length', type=int, default=2048, help="Maximum sequence length, default is 2048. We auto support RoPE Scaling internally!")
    model_group.add_argument('--dtype', type=str, default=None, help="Data type for model (None for auto detection)")
    model_group.add_argument('--load_in_4bit', action='store_true', help="Use 4bit quantization to reduce memory usage")
    """

    model_name: str = "unsloth/llama-3-8b"  # Model name to load
    max_seq_length: int = (
        2048  # Maximum sequence length, default is 2048. We auto support RoPE Scaling internally!
    )
    dtype: str = None  # Data type for model (None for auto detection)
    load_in_4bit: bool = True  # Use 4bit quantization to reduce memory usage


class LoraArgs(BaseModel):
    """
    lora_group = parser.add_argument_group("🧠 LoRA Options", "These options are used to configure the LoRA model.")
    lora_group.add_argument('--r', type=int, default=16, help="Rank for Lora model, default is 16.  (common values: 8, 16, 32, 64, 128)")
    lora_group.add_argument('--lora_alpha', type=int, default=16, help="LoRA alpha parameter, default is 16. (common values: 8, 16, 32, 64, 128)")
    lora_group.add_argument('--lora_dropout', type=float, default=0, help="LoRA dropout rate, default is 0.0 which is optimized.")
    lora_group.add_argument('--bias', type=str, default="none", help="Bias setting for LoRA")
    lora_group.add_argument('--use_gradient_checkpointing', type=str, default="unsloth", help="Use gradient checkpointing")
    lora_group.add_argument('--random_state', type=int, default=3407, help="Random state for reproducibility, default is 3407.")
    lora_group.add_argument('--use_rslora', action='store_true', help="Use rank stabilized LoRA")
    lora_group.add_argument('--loftq_config', type=str, default=None, help="Configuration for LoftQ")
    """

    r: int = (
        16  # Rank for Lora model, default is 16.  (common values: 8, 16, 32, 64, 128)
    )
    lora_alpha: int = (
        16  # LoRA alpha parameter, default is 16. (common values: 8, 16, 32, 64, 128)
    )
    lora_dropout: float = 0  # LoRA dropout rate, default is 0.0 which is optimized.
    bias: str = "none"  # Bias setting for LoRA
    use_gradient_checkpointing: str = "unsloth"  # Use gradient checkpointing
    use_rslora: bool = False  # Use rank stabilized LoRA
    loftq_config: str = None  # Configuration for LoftQ
    random_state: int = 3407  # Random state for reproducibility, default is 3407.


class TrainingArgs(BaseModel):
    """
    training_group = parser.add_argument_group("🎓 Training Options")
    training_group.add_argument('--per_device_train_batch_size', type=int, default=2, help="Batch size per device during training, default is 2.")
    training_group.add_argument('--gradient_accumulation_steps', type=int, default=4, help="Number of gradient accumulation steps, default is 4.")
    training_group.add_argument('--warmup_steps', type=int, default=5, help="Number of warmup steps, default is 5.")
    training_group.add_argument('--max_steps', type=int, default=400, help="Maximum number of training steps.")
    training_group.add_argument('--learning_rate', type=float, default=2e-4, help="Learning rate, default is 2e-4.")
    training_group.add_argument('--optim', type=str, default="adamw_8bit", help="Optimizer type.")
    training_group.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay, default is 0.01.")
    training_group.add_argument('--lr_scheduler_type', type=str, default="linear", help="Learning rate scheduler type, default is 'linear'.")
    training_group.add_argument('--seed', type=int, default=3407, help="Seed for reproducibility, default is 3407.")
    """

    per_device_train_batch_size: int = (
        2  # Batch size per device during training, default is 2.
    )
    gradient_accumulation_steps: int = (
        4  # Number of gradient accumulation steps, default is 4.
    )
    warmup_steps: int = 5  # Number of warmup steps, default is 5.
    max_steps: int = -1  # Maximum number of training steps.
    num_train_epochs: int = 1  # Number of training epochs, default is 1.
    learning_rate: float = 2e-4  # Learning rate, default is 2e-4.
    optim: str = "adamw_8bit"  # Optimizer type.
    weight_decay: float = 0.01  # Weight decay, default is 0.01.
    lr_scheduler_type: str = (
        "linear"  # Learning rate scheduler type, default is 'linear'.
    )
    seed: int = 3407  # Seed for reproducibility, default is 3407.


class ReportArgs(BaseModel):
    """
    # Report/Logging arguments
    report_group = parser.add_argument_group("📊 Report Options")
    report_group.add_argument('--report_to', type=str, default="tensorboard",
        choices=["azure_ml", "clearml", "codecarbon", "comet_ml", "dagshub", "dvclive", "flyte", "mlflow", "neptune", "tensorboard", "wandb", "all", "none"],
        help="The list of integrations to report the results and logs to. Supported platforms are: \n\t\t 'azure_ml', 'clearml', 'codecarbon', 'comet_ml', 'dagshub', 'dvclive', 'flyte', 'mlflow', 'neptune', 'tensorboard', and 'wandb'. Use 'all' to report to all integrations installed, 'none' for no integrations.")
    report_group.add_argument('--logging_steps', type=int, default=1, help="Logging steps, default is 1")
    """

    report_to: str = "tensorboard"  # Report to tensorboard
    logging_steps: int = 1  # Logging steps, default is 1


class SaveArgs(BaseModel):
    """
    # Saving arguments
    save_group = parser.add_argument_group('💾 Save Model Options')
    save_group.add_argument('--output_dir', type=str, default="outputs", help="Output directory")
    save_group.add_argument('--save_model', action='store_true', help="Save the model after training")
    save_group.add_argument('--save_method', type=str, default="merged_16bit", choices=["merged_16bit", "merged_4bit", "lora"], help="Save method for the model, default is 'merged_16bit'")
    save_group.add_argument('--save_gguf', action='store_true', help="Convert the model to GGUF after training")
    save_group.add_argument('--save_path', type=str, default="model", help="Path to save the model")
    save_group.add_argument('--quantization', type=str, default="q8_0", nargs="+",
        help="Quantization method for saving the model. common values ('f16', 'q4_k_m', 'q8_0'), Check our wiki for all quantization methods https://github.com/unslothai/unsloth/wiki#saving-to-gguf ")
    """

    output_dir: str = "outputs"  # Output directory
    save_model: bool = True  # Save the model after training
    save_method: str = (
        "merged_16bit"  # Save method for the model, default is 'merged_16bit'
    )
    save_gguf: bool = True  # Convert the model to GGUF after training
    save_path: str = "model"  # Path to save the model
    quantization: str = (
        "q4_k_m"  # Quantization method for saving the model. common values ('f16', 'q4_k_m', 'q8_0'), Check our wiki for all quantization methods https://github.com/unslothai/unsloth/wiki#saving-to-gguf
    )


class PushArgs(BaseModel):
    """
    # Push arguments
    push_group = parser.add_argument_group('🚀 Push Model Options')
    push_group.add_argument('--push_model', action='store_true', help="Push the model to Hugging Face hub after training")
    push_group.add_argument('--push_gguf', action='store_true', help="Push the model as GGUF to Hugging Face hub after training")
    push_group.add_argument('--hub_path', type=str, default="hf/model", help="Path on Hugging Face hub to push the model")
    push_group.add_argument('--hub_token', type=str, help="Token for pushing the model to Hugging Face hub")
    """

    push_model: bool = False  # Push the model to Hugging Face hub after training
    push_gguf: bool = False  # Push the model as GGUF to Hugging Face hub after training
    hub_path: str = "hf/model"  # Path on Hugging Face hub to push the model
    hub_token: str = None  # Token for pushing the model to Hugging Face hub


class Config(BaseModel):
    """config for finetune unsloth"""

    model: ModelArgs = ModelArgs()
    lora: LoraArgs = LoraArgs()
    training: TrainingArgs = TrainingArgs()
    report: ReportArgs = ReportArgs()
    save: SaveArgs = SaveArgs()
    push: PushArgs = PushArgs()
