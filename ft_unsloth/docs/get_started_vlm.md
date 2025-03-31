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
mkdir -p mount_data/vlm
```

## prepare the dataset
copy the dataset to the mount dir `mount_data/vlm`
```bash
cp -r /path/to/your/dataset/mount_data/vlm
```
use toy dataset for testing
```bash
cp -r ft_unsloth/toy_datasets/vlm mount_data/vlm/train_data
```

## copy config file
```bash
cp ft_unsloth/configs/vlm-example.yaml mount_data/vlm/config.yaml
```

## configure the config file `mount_data/vlm/config.yaml`
See the example config file `ft_unsloth/configs/vlm-example.yaml` or `ft_unsloth/config.py` for more details.

## run the finetune
```bash
python ft_unsloth/train.py --config mount_data/vlm/config.yaml --mode vlm
```






