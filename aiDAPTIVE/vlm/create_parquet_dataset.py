from datasets import Dataset, Image
import os
import json
import math
import argparse

def create_image_text_dataset(image_dir, jsonl_file):
    dataset_dict = {
        "question": [],
        "answer": [],
        "image_path": [],
        "image": []
    }
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            image_path = os.path.join(image_dir, item['file_name'])
            if os.path.exists(image_path):
                try:
                    dataset_dict["question"].append(item["instruction"])
                    dataset_dict["answer"].append(item['text'])
                    dataset_dict["image"].append(image_path)
                    dataset_dict["image_path"].append(image_path)
                except Exception as e:
                    print(f"error {image_path}: {e}")
                    continue
    dataset = Dataset.from_dict(dataset_dict)
    dataset = dataset.cast_column("image", Image())
    return dataset

def save_sharded_parquet(file_name, dataset, output_dir, num_shards):
    # Split dataset into `num_shards` and save each shard as a separate Parquet file
    shard_size = math.ceil(len(dataset) / num_shards)
    os.makedirs(output_dir, exist_ok=True)
    for shard_idx in range(num_shards):
        start_idx = shard_idx * shard_size
        end_idx = min((shard_idx + 1) * shard_size, len(dataset))
        shard = dataset.select(range(start_idx, end_idx))  # Select subset of the dataset
        # Save shard as a Parquet file
        shard_path = os.path.join(output_dir, f"{file_name}-{str(shard_idx).zfill(5)}-of-{str(num_shards).zfill(5)}.parquet")
        shard.to_parquet(shard_path)
        print(f"Saved shard {shard_idx} to {shard_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Create and save a dataset with images in Parquet format.")
    parser.add_argument(
        "--dataset_dir", type=str, default="./example_dataset",help="dataset directory")
    parser.add_argument(
        "--num_shards",  type=int, default=1, help="Number of shards to split the dataset into (default is 1)")
    parser.add_argument(
        "--parquet_name", type =str, default="train", help="File name will be {file_name}-0000N-of-0000Y.parquet") 
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    IMAGE_DIR = os.path.join(args.dataset_dir, "image")
    JSONL_FILE = os.path.join(args.dataset_dir,"metadata.jsonl")
    RESULT_DIR = os.path.join(args.dataset_dir, "data")
    PARQUET_NAME = args.parquet_name
    # Create dataset
    dataset = create_image_text_dataset(IMAGE_DIR, JSONL_FILE)
    # Save as sharded Parquet files
    save_sharded_parquet(PARQUET_NAME, dataset, RESULT_DIR, num_shards=1)
