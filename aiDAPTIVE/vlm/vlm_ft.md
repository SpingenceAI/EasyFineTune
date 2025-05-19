## fine-tune vlm model

1. modify training config

```bash
phison_config/
  - env_config.yaml
    - model_name_or_path: path to vlm model (/home/root/workspace/Qwen2-VL-2B-Instruct)
    - data_path: path to dataset (dataset_config.yaml)
  - exp_config.yaml (hyper parameter)
  - dataset_config.yaml
    - data_path: path to dataset (dataset_config.yaml)
```

2. start training

```bash
phisonai2 --env_config phison_config/env_config.yaml --exp_config phison_config/exp_config.yaml
```
