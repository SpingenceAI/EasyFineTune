# Quantization using llama.cpp

## Build the docker container

```bash
make build_quantization_llama_cpp
```



## create mount dir
```bash
mkdir -p mount_data/quantization
```
download model to the mount dir `mount_data/quantization`


## prepare config file

copy the config file to the mount dir
```bash
cp quantization/configs/llama_cpp-example.yaml mount_data/quantization/llama-cpp-config.yaml
```

modify the config file
```yaml
model_name: "mount_data/quantization/Qwen2-1.5B" # model path under mount_data/quantization
quantization: "Q8_0" # Q4_0, Q4_K_M, Q8_0, Q8_K_M
```



## Launch the docker container

```bash
make launch_quantization_llama_cpp
```



execute the quantization command
```bash
python3 llama_cpp_quantization.py --config /workspace/mount_data/quantization/llama-cpp-config.yaml
```

model will be save in the `save_path` directory `mount_data/quantization/quantized_model`







