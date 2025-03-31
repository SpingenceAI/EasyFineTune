# Generate training data

## Get Started (under `EasyFineTune` directory)

1. Create `mount_data/generate_data` directory if not exists
   ```bash
   mkdir -p mount_data/generate_data
   ```
2. copy `generate_data/configs/config-example.yaml` to `mount_data/generate_data/config.yaml`
   ```bash
   # example
   cp generate_data/configs/config-example.yaml mount_data/generate_data/config.yaml
   ```
3. setup `mount_data/generate_data/config.yaml`, providers [litellm supported providers](https://docs.litellm.ai/docs/providers)

   OpenAI:

   ```yaml
   # mount_data/generate_data/config.yaml (if using openai)
   llm:
     model: openai/gpt-4o-mini
     api_key: ENTER_YOUR_OPENAI_API_KEY
     base_url: null
   ```

   Ollama:

   ```yaml
   # mount_data/generate_data/config.yaml (if using ollama)
   llm:
     model: ollama/llama3.1:8b
     base_url: http://localhost:11434
   ```

4. prepare documents to `mount_data/generate_data/documents`
   ```bash
   # example
   cp -r generate_data/toy_datasets mount_data/generate_data/documents
   ```
   documents structure:
   ```bash
   documents/
   ├── document_1.txt
   ├── document_2.pdf
   └── document_3.docx
   ```
5. Build docker image
   ```bash
   make build_generate_data
   ```
6. Launch the container
   ```bash
   make launch_generate_data
   ```
7. Run the following command to generate training data
   ```bash
   python generate_data/run.py \
    --config mount_data/generate_data/config.yaml \
    --input mount_data/generate_data/documents \
    --output mount_data/generate_data/output
   ```
8. check the generated data in `mount_data/generate_data/output/train_data.json`, should have a list of json objects, each object contains `instruction`, `input`, `output`
