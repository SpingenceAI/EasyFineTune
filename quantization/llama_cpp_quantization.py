import argparse
import yaml
import subprocess
import os
from pydantic import BaseModel
from typing import Optional
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class QuantizationArgs(BaseModel):
    model_name: Optional[str] = None  # Name of the model
    quantization: Optional[str] = None  # Quantization method
    save_path: Optional[str] = None  # Save path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    return parser.parse_args()


def load_config(config_path: str) -> QuantizationArgs:
    """load config from yaml file"""
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return QuantizationArgs(**config)


def run_command_with_live_output(cmd: list, work_dir: str):
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True,
        cwd=work_dir,
    )

    remaining_output, remaining_error = process.communicate()
    if remaining_output:
        logger.info(remaining_output.strip())
    if remaining_error:
        logger.error(remaining_error.strip())

    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)


def hf_to_gguf(model_path: str, quantization: str = None, save_path: str = None):
    """convert the model to gguf format"""
    model_path = os.path.abspath(model_path)
    save_path = os.path.abspath(save_path)

    cmd = ["python3", "convert_hf_to_gguf.py", model_path]
    work_dir = f"/workspace/llama.cpp"
    run_command_with_live_output(cmd, work_dir)
    if not quantization:
        pass
    else:
        quantization = quantization.upper()
        if quantization not in [
            "Q4_0",
            "Q4_K_M",
            "Q4_K_L",
            "Q4_K_S",
            "Q8_0",
            "Q8_K_M",
            "Q8_K_L",
            "Q8_K_S",
        ]:
            raise ValueError(f"Invalid quantization method: {quantization}")
        model_name = os.path.basename(model_path)

        # save dir for gguf f16 and gguf quantization
        save_dir = os.path.join(save_path, f"llama_cpp_quantized_model-{quantization}")
        os.makedirs(save_dir, exist_ok=True)

        # f16 model path
        f16_model_path = os.path.join(save_dir, f"{model_name}-F16.gguf")

        # quantization model path
        quantize_model_path = os.path.join(
            save_dir, f"{model_name}-{quantization}.gguf"
        )

        # quantization command
        cmd = ["./llama-quantize", f16_model_path, quantize_model_path, quantization]
        run_command_with_live_output(cmd, work_dir)


def main():
    args = parse_args()
    config = load_config(args.config)
    MODEL_ID = config.model_name
    QUANTIZATION = config.quantization
    SAVE_PATH = config.save_path
    hf_to_gguf(MODEL_ID, QUANTIZATION, SAVE_PATH)


if __name__ == "__main__":
    main()
