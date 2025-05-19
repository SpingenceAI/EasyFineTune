# aiDaptiv vlm fine-tune:

## create custum dataset
- data structure:
```bash
dataset
  - image
    - image01.jpg
    - image02.jpg
    -...
  - metadeta.jsonl 
```

- metadata_example:
```jsonl
{"file_name": "710_4.jpg", "instruction": "What model is this computer?", "text": "MIC-710AIL"}
{"file_name": "710_2.jpg", "instruction": "What model is this computer?", "text": "MIC-710AIL"}
{"file_name": "710_1.jpg", "instruction": "What model is this computer?", "text": "MIC-710AIL"}
...
```

- create parquet file:
```bash
cd ./vlm_ft
python3 create_parquet_dataset.py --dataset_dir /path/to/dataset

*data key in parquet file will be:
  question_key: "question"
  answer_key: "answer"
  image_key: "image"
  label_key: "answer"
```


## fine-tune vlm model
1. modify training config
```bash
phison_config
  - env_config.yaml << model path, dataset config path
  - exp_config.yaml << hyper parameter
  - VQA_dataset_config.yaml << dataset path and parsing
```

2. start training
```bash
cd ./vlm_ft
phisonai2 --env_config phison_config/env_config.yaml --exp_config phison_config/exp_config.yaml
```


