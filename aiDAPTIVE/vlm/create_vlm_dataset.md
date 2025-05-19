# Create VLM dataset

dataset format should be parquet file

## prepare dataset for parquet file
File structure:
```
example_dataset/
├── image/ xxx.jpg
└── metadata.jsonl
```

metadata.jsonl format:
```jsonl
{"file_name": "710_4.jpg", "instruction": "What model is this computer?", "text": "MIC-710AIL"}
{"file_name": "710_2.jpg", "instruction": "What model is this computer?", "text": "MIC-710AIL"}
{"file_name": "710_1.jpg", "instruction": "What model is this computer?", "text": "MIC-710AIL"}
{"file_name": "710_2.jpg", "instruction": "What model is this computer?", "text": "MIC-710AIL"}
{"file_name": "710_3.jpg", "instruction": "What model is this computer?", "text": "MIC-710AIL"}
{"file_name": "733_1.jpg", "instruction": "What model is this computer?", "text": "MIC-733AO"}
{"file_name": "733_2.jpg", "instruction": "What model is this computer?", "text": "MIC-733AO"}
{"file_name": "733_3.jpg", "instruction": "What model is this computer?", "text": "MIC-733AO"}
```


## create dataset

```bash
python create_vlm_dataset.py --dataset_dir ./example_dataset --num_shards 1 --parquet_name train
```

```python
*data key in parquet file will be:
  question_key: "question"
  answer_key: "answer"
  image_key: "image"
  label_key: "answer"
```

output data will be saved in example_dataset/data/train-00000-of-00001.parquet
