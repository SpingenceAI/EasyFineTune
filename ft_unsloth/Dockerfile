FROM nvcr.io/nvidia/pytorch:25.03-py3
WORKDIR /workspace

RUN pip install --no-deps bitsandbytes accelerate xformers==0.0.29 peft trl triton
RUN pip install --no-deps cut_cross_entropy unsloth_zoo
RUN pip install sentencepiece protobuf datasets huggingface_hub hf_transfer
RUN pip install --no-deps unsloth
RUN pip install pydantic
RUN pip install transformers==4.47.1


RUN git clone https://github.com/ggml-org/llama.cpp.git
RUN apt-get update
RUN apt-get install -y cmake
WORKDIR /workspace/llama.cpp
RUN cmake -B build
RUN cmake --build build --config Release
RUN cp build/bin/llama-quantize /workspace/llama.cpp/llama-quantize
RUN pip install --no-deps datasets==3.5.0
WORKDIR /workspace