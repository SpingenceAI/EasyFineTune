# Get started with LLM finetune using Unsloth

## Build the docker container

```bash
make build_ft_unsloth
```

## Launch the docker container

```bash
make launch_ft_unsloth
```

## create mount dir
```bash
mkdir -p mount_data/llm
```

## prepare the dataset
copy the dataset to the mount dir `mount_data/llm`
use generated dataset for finetune
```bash
cp -r mount_data/generate_data/output/train_data.json mount_data/llm/train_data.json
```
use toy dataset for testing
```bash
cp -r ft_unsloth/toy_datasets/llm/train_data.json mount_data/llm/train_data.json
```

## copy config file
```bash
cp ft_unsloth/configs/llm-example.yaml mount_data/llm/config.yaml
```

## configure the config file `mount_data/llm/config.yaml`
See the example config file `ft_unsloth/configs/llm-example.yaml` or `ft_unsloth/config.py` for more details.

## run the finetune
```bash
python ft_unsloth/train.py --config mount_data/llm/config.yaml --mode llm
```






