import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--mode", type=str, default="llm")
    return parser.parse_args()


def main():
    args = parse_args()

    mode = args.mode
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file {args.config} not found")
    if mode == "llm":
        from llm import train

        train(args.config)
    elif mode == "vlm":
        from vlm import train

        train(args.config)
    else:
        raise ValueError(f"Invalid mode: {mode}")


if __name__ == "__main__":
    main()
