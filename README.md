# combine_sharegpt.py

This project provides a Python script to combine, preprocess, and filter ShareGPT data from multiple JSON files. It includes features such as embedding computation, deduplication, and histogram normalization.

## Features

- Combines multiple ShareGPT JSON files
- Preprocesses conversations (removes GPT-isms, validates format)
- Computes embeddings for question-answer pairs
- Deduplicates based on cosine similarity
- Normalizes data distribution
- Configurable via YAML file

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- NumPy
- SciPy
- scikit-learn
- YAML

You can install the required packages using the provided `requirements.txt` file:

```
pip install -r requirements.txt
```

## Usage

1. Place your ShareGPT JSON files in the input directory specified in the configuration file.
2. Adjust the `combine_sharegpt_config.yaml` file to your needs.
3. Run the script:

```
python combine_sharegpt.py [path_to_config_file]
```

If no config file is specified, it will use `combine_sharegpt_config.yaml` in the same directory by default.

## Configuration Details

The `combine_sharegpt_config.yaml` file contains various settings for the script. Here's a detailed explanation of each configuration section:

### Embedding

- `tokenizer_name`: The name of the tokenizer to use for text processing. Default is "bert-base-uncased".
- `max_length`: Maximum number of tokens to process in a single pass. Default is 8192.
- `embedding_model_path`: Local or HuggingFace path to the embedding model directory.
- `embedding_model_name`: Name of the embedding model to use. Default is "nomic-embed-text-v1.5".
- `trust_remote_code`: Whether to trust remote code when loading the model. Default is true.
- `safe_serialization`: Use safe serialization when loading the model. Default is true.
- `rotary_scaling_factor`: Scaling factor for rotary embeddings. Default is 2.
- `local_files_only`: Load model files from the local file system only. Default is true.
- `initial_batch_size`: Initial batch size for embedding computation. Default is 2000.
- `memory_fraction`: Fraction of device memory to use for batching. Default is 0.7.

### Filtering

- `max_output_size`: Maximum number of output records after filtering and preprocessing. Default is 64000.
- `dedup`: Whether to deduplicate the input data using embeddings before processing. Default is true.
- `norm`: Whether to normalize the distribution of the combined data set length. Default is true.
- `similarity_threshold`: Threshold for similarity to consider pairs as duplicates (0-1). Default is 0.95.
- `F1`: First filtering method. Default is "cosine".
- `F2`: Second filtering method. Default is "kl".
- `filter_batch_size`: Batch size for PyTorch cosine similarity calculation. Default is 1000.
- `force_num_bins`: Set to an integer to override automatic bin selection. Default is null.

### Input

- `directory`: Directory containing input sharegpt.json files. Default is "./".

### Output

- `directory`: Directory to save output files. Default is "./".
- `train_output_file`: Base name for train output files. Default is "train_sharegpt".
- `test_output_file`: Base name for test output files. Default is "test_sharegpt".
- `save_json`: Whether to save output in JSON format. Default is true.
- `save_jsonl`: Whether to save output in JSONL format. Default is true.

### Logging

- `level`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default is "INFO".
- `format`: Log message format. Default is "%(asctime)s - %(name)s - %(levelname)s - %(message)s".

### Preprocessing

- `system_prompt`: Default system prompt to use if none is provided in the input data.
- `gpt_strip_out`: List of phrases to strip out from GPT responses. This helps remove common filler phrases and improve the quality of the data.
- `rejection_phrases`: List of phrases in the GPT response that cause a conversation to be rejected. This helps filter out responses where the AI model is explaining its limitations or apologizing.

## Customizing the Configuration

You can customize these settings to fit your specific needs:

1. Adjust the embedding model and tokenizer if you want to use a different one.
2. Modify the filtering parameters to change how strict the deduplication and normalization processes are.
3. Change the input and output directories to match your file structure.
4. Adjust the preprocessing settings to refine how the conversations are cleaned and filtered.

Remember to test your changes on a small subset of your data before processing large datasets to ensure the results meet your expectations.

## Output

The script generates the following output files:

- `train_sharegpt.json` and `train_sharegpt.jsonl`: Training data
- `test_sharegpt.json` and `test_sharegpt.jsonl`: Test data
- `total_sharegpt.json` and `total_sharegpt.jsonl`: Combined data
