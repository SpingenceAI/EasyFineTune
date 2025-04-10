# Quantization using vllm (PQT)

## Build the docker container

```bash
make build_quantization_vllm
```

## create mount dir

```bash
mkdir -p mount_data/quantization
```

download model to the mount dir `mount_data/quantization`

## prepare config file

copy the config file to the mount dir

```bash
cp quantization/configs/vllm-example.yaml mount_data/quantization/vllm-config.yaml
```

modify the config file

```yaml
model_name: 'Qwen/Qwen2-1.5B' # huggingface model id
```

## vllm quantization

launch the docker container

```bash
make launch_quantization_vllm
```

## Login huggingface

```bash
git config --global credential.helper store
huggingface-cli login
```

execute the quantization script

```bash
python3 quantization/vllm_quantization.py --config /workspace/mount_data/quantization/vllm-config.yaml
```
