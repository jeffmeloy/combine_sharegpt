# combine_sharegpt.py

This Python script provides a comprehensive pipeline for processing, filtering, embedding, and saving conversational data in the ShareGPT format. It's designed to help you build high-quality datasets for training or evaluating conversational AI models.

## Features

- **Data Loading & Combining:**  Combines multiple ShareGPT JSON files.
- **Preprocessing:**
    - Cleans up conversations, removes common GPT-generated phrases, and validates data formats.
    - Standardizes roles (e.g., "user" to "human", "assistant" to "gpt"). 
    - Filters out conversations containing rejection phrases (customizable) or insufficient code/ASCII characters (for code-related datasets).
    - Ensures conversations have valid and standardized system prompts.
- **Embedding Generation:**
    - Computes embeddings for question-answer pairs using state-of-the-art models like `nomic-embed-text-v1.5`, which supports long sequences and adjustable dimensionality.
    - Optionally stores embeddings in a [ChromaDB](https://www.trychroma.com/) collection for efficient similarity search and analysis.
- **Filtering & Deduplication:**
    - Removes duplicate conversations based on cosine similarity.
    - Uses advanced techniques like cosine similarity, KL-divergence, and histogram-based filtering to select the most diverse and informative conversations. 
    - Provides multiple filtering options and allows you to customize the filtering thresholds.
- **System Prompt Improvement:**
    - Integrates with the `llama.cpp` language model to automatically analyze conversations and generate potentially improved system prompts, leading to higher-quality interactions in your dataset.
- **Data Normalization & Splitting:**
    - Normalizes the data distribution to ensure a balanced representation of different conversation sizes.
    - Automatically splits data into training and test sets.
- **Output & Logging:** 
    - Saves processed data in JSON or JSON Lines (JSONL) format.
    - Provides detailed logging for monitoring progress and debugging.

## Requirements

- Python 3.7+
- PyTorch 
- Transformers
- NumPy
- SciPy
- scikit-learn
- YAML
- [ChromaDB](https://docs.trychroma.com/getting-started) (optional, for embedding storage)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) (optional, for system prompt improvement)

You can install the Python dependencies using the provided `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```
 
## Usage

Prepare Your Data: Place your ShareGPT JSON files in the input directory specified in the configuration file (combine_sharegpt_config.yaml).

Configure: Customize the settings in combine_sharegpt_config.yaml to match your dataset, embedding model, filtering preferences, and other options. See the Configuration Details section below for detailed explanations.

    Run: Execute the script:

    ```bash     
    python combine_sharegpt.py [path_to_config_file]
    ```

If no config file path is provided, the script will use combine_sharegpt_config.yaml in the current directory by default.

## Configuration Details

The combine_sharegpt_config.yaml file controls the behavior of the pipeline. Refer to the example configuration file and comments within it for explanations of the various settings. The configuration is organized into the following sections:
- embedding: Settings for embedding generation, including the model, tokenizer, batch size, and memory management.
- filtering: Settings for the optional deduplication, similarity thresholds, and the chosen filtering methods.
- input: Specifies the input data directory and any unmodified files to include.
- output: Controls output data formats, directory, and file naming.
- chromadb: Configuration for the optional ChromaDB integration (embedding storage and retrieval).
- system_prompt: Settings for the optional llama.cpp integration to improve system prompts.
- logging: Configures the logging level and format.
- preprocessing: Settings for data cleaning, including default system prompts, phrases to strip from GPT responses, and rejection phrases.

## Output

The script generates the following output files:
- Training Data: train_sharegpt.json and train_sharegpt.jsonl
- Test Data: test_sharegpt.json and test_sharegpt.jsonl
- Combined Data: total_sharegpt.json and total_sharegpt.jsonl

The specific formats (JSON or JSONL) and file names can be customized in the configuration file.

## Example Configuration:

Refer to the combine_sharegpt_config.yaml file in this repository for a detailed example configuration.

## Code Documentation

This section provides documentation for each function in the `combine_sharegpt.py` script including descriptions of their inputs, outputs, and processing logic.

**Contents**
- `load_config(config_path: str) -> dict`
- `SingleLineHandler(logging.StreamHandler)`
- `load_files(config: dict) -> list[dict]`
- `is_valid_conversation(conversation: list[dict], config: dict) -> Tuple[bool, list[dict]]`
- `strip_gptisms(conversation: list[dict], config: dict) -> list[dict]` 
- `strip_non_code_and_ascii(sharegpt_data: list[dict], config: dict) -> list[dict]` 
- `preprocess_data(sharegpt_data: list[dict], config: dict) -> list[dict]`
- `get_qa_pairs(data: list[dict]) -> list[Tuple[str, str]]`
- `get_embeddings(texts: list[str], model: torch.nn.Module, tokenizer: PreTrainedTokenizer) -> Tuple[torch.Tensor, int]`
- `get_qa_pair_embeddings(qa_pairs: list[Tuple[str, str]], model: torch.nn.Module, max_length: int, model_max_length: int) -> Tuple[np.ndarray, int]`
- `compute_embeddings(sharegpt_data: list[dict], config: dict) -> list[dict]`
- `get_optimal_bins(sharegpt_data: list[dict], config: dict) -> Tuple[int, np.ndarray]`
- `deduplication(sharegpt_data: list[dict], config: dict, bin_indices: np.ndarray) -> list[dict]`
- `histogram_filter(sharegpt_data: list[dict], nbins: int, bin_indices: np.ndarray, config: dict) -> list[dict]`
- `normalize_sharegpt(sharegpt_data: list[dict], nbins: int, config: dict) -> list[dict]`
- `save_chromaDB(sharegpt_data: list[dict], config: dict) -> None`
- `save_json_file(sharegpt_data: list[dict], output_file_json: str) -> None`
- `save_jsonl_file(sharegpt_data: list[dict], output_file_jsonl: str) -> None`
- `save_sharegpt(sharegpt_data: list[dict], nbins: int, config: dict) -> None`
- `truncate_qa(question: str, answer: str, max_length: int, tokenizer: PreTrainedTokenizer) -> Tuple[str, str]`
- `strip_out_rejection_phrases(sharegpt_data: list[dict], config: dict) -> list[dict]` 
- `mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor`
- `check_system_prompt(conversation: list[dict], config: dict) -> list[dict]`
- `plot_histogram(sharegpt_data: list[dict], nbins: int) -> None`
- `plot_normalized_histogram(normalized_data: list[dict], nbins: int) -> None`
- `initialize_llama(config: dict) -> list[str]`
- `improve_system_prompt(sharegpt_data: list[dict], config: dict) -> list[dict]`
- `get_cosine_scores(qa_pair_embeddings: np.ndarray, filter_batch_size: int) -> np.ndarray` 
- `get_filtered_indices(qa_pair_embeddings: np.ndarray, filter_ratio: float, config: dict) -> np.ndarray`
- `sort_and_truncate(sharegpt_data: list[dict], config: dict) -> list[dict]`
- `append_file_to_sharegpt(sharegpt_data: list[dict], json_file: str, config: dict) -> list[dict]`
- `combine_sharegpt(config_file: str) -> None`
- `main() -> None`

### `load_config(config_path: str) -> dict`

Loads the configuration settings from a YAML file.

**Inputs:**

- `config_path`: Path to the YAML configuration file.

**Outputs:**

- A dictionary containing the loaded configuration settings.

**Processing Logic:**

1. Opens the YAML file specified by `config_path`.
2. Uses the `yaml.safe_load()` function to parse the YAML content into a Python dictionary. 
3. Returns the loaded configuration dictionary.

###  `SingleLineHandler(logging.StreamHandler)`

A custom logging handler that overwrites the previous log line on the console, providing cleaner progress updates.

**(Inherits from `logging.StreamHandler`)** 

**Key Method:**

- `emit(self, record)`: Overrides the default `emit` method of `StreamHandler`. Formats the log record and writes it to the console, overwriting the previous line.

### `load_files(config: dict) -> list[dict]`

Loads and combines data from multiple JSON and JSONL files in the input directory specified in the configuration.

**Inputs:**

- `config`: The loaded configuration dictionary.

**Outputs:**

- A list of dictionaries, where each dictionary represents a single data entry from the combined JSON files.

**Processing Logic:**

1. Retrieves the input directory path from the `config` dictionary.
2. Gets a list of all JSON and JSONL files in the input directory.
3. Iterates through each JSON and JSONL file:
    - Opens the file.
    - Loads the file content. 
    - Extends the main data list with the loaded data.
4. Returns the combined data list.

###  `is_valid_conversation(conversation: list[dict], config: dict) -> Tuple[bool, list[dict]]`

Checks if a conversation meets the following criteria:
- Contains both "human" and "gpt" roles.
- Has a valid system prompt (if required by the configuration).
- Follows a valid conversational flow (human message before GPT response).

It also standardizes role names from "user"/"assistant" to "human"/"gpt".

**Inputs:**

- `conversation`:  A list of dictionaries representing a single conversation. Each dictionary should have `from` and `value` keys.
- `config`: The loaded configuration dictionary. 

**Outputs:**

- `is_valid`: A boolean value indicating whether the conversation is valid.
- `processed_conversation`:  The processed conversation with standardized roles and potentially a modified system prompt.

**Processing Logic:**

1. **Role Standardization:**  Replaces "user" with "human" and "assistant" with "gpt" in the `from` keys of the conversation.
2. **System Prompt Check:** (If enabled in `config`):
    - Checks if the first message is a system message and updates it if needed. 
    - Inserts a default system message if none is present.
3. **Validity Checks:**
    - Checks if both "human" and "gpt" roles are present.
    - Ensures that the conversation flow is valid (human before GPT).
4. Returns `True` and the `processed_conversation` if valid, otherwise `False` and the original conversation.

### `strip_gptisms(conversation: list[dict], config: dict) -> list[dict]` 

Removes common GPT-generated phrases from the conversation.

**Inputs:**

- `conversation`: A list of dictionaries representing the conversation.
- `config`: The loaded configuration dictionary.

**Outputs:**

- A list of dictionaries representing the conversation with GPT-isms removed. 

**Processing Logic:**

1. Iterates through each message in the conversation.
2. If the message is from "gpt", it removes any phrases listed in the `gpt_strip_out` section of the configuration file.

### `strip_non_code_and_ascii(sharegpt_data: list[dict], config: dict) -> list[dict]`

Removes conversations that do not contain a sufficient ratio of ASCII and code-related characters (useful for code-focused datasets).

**Inputs:**

- `sharegpt_data`: The list of conversations. 
- `config`: The loaded configuration dictionary.

**Outputs:**

- A list of conversations where conversations with insufficient ASCII or code characters have been removed.

**Processing Logic:**

1. Iterates through each conversation.
2. For each GPT message:
    - Calculates the ratio of valid characters (ASCII + code characters).
    - If the ratio is below the threshold specified in the config, the conversation is removed.

### `preprocess_data(sharegpt_data: list[dict], config: dict) -> list[dict]`

Orchestrates the preprocessing steps for the entire dataset.

**Inputs:** 

- `sharegpt_data`: The loaded list of conversations.
- `config`: The loaded configuration dictionary.

**Outputs:**

- A list of dictionaries representing the preprocessed conversations.

**Processing Logic:**

1. **Conversation Validation:** Calls `is_valid_conversation` to validate each conversation and standardize roles.
2. **GPT-ism Removal:** Calls `strip_gptisms` to remove common GPT phrases. 
3. **Non-Code/ASCII Removal:** Calls `strip_non_code_and_ascii` (if enabled).
4. **Size Calculation:**  Calculates the size (`nbytes`) of each conversation.
5. **Sorting and Truncation:** Sorts conversations by size and truncates the dataset if it exceeds the maximum size specified in the configuration. 
6. **Rejection Phrase Removal:** Removes conversations containing predefined rejection phrases from the GPT model (like apologies or limitations).
7. **Exact Duplicate Removal:** Removes identical conversations.
8. Returns the preprocessed dataset.

###  `get_qa_pairs(data: list[dict]) -> list[Tuple[str, str]]`

Extracts question-answer pairs from a list of conversations.

**Inputs:**

- `data`: A list of conversation dictionaries, where each dictionary represents a single conversation.

**Outputs:**

- `qa_pairs`: A list of tuples, where each tuple contains a (question, answer) pair.

**Processing Logic:**

1. Iterates through each conversation in the `data` list.
2. Within each conversation:
    - Collects all messages from the "human" role into `question_parts`.
    - Collects all messages from the "gpt" role into `answer_parts`.
3. Joins the `question_parts` with " [SEP] " and the `answer_parts` with " [SEP] " to create the full question and answer strings.
4. Prepends "search_query: " to both the question and answer strings. 
5. Appends the (question, answer) tuple to the `qa_pairs` list.
6. Returns the list of extracted question-answer pairs.

### `get_embeddings(texts: list[str], model: torch.nn.Module, tokenizer: PreTrainedTokenizer) -> Tuple[torch.Tensor, int]`

Computes embeddings for a list of text strings using the specified embedding model and tokenizer.

**Inputs:**

- `texts`: A list of text strings to embed.
- `model`: The loaded embedding model (e.g., from Hugging Face Transformers).
- `tokenizer`: The tokenizer associated with the embedding model.

**Outputs:**

- `embeddings`: A PyTorch tensor containing the computed embeddings for the input texts.
- `max_mem`: The maximum GPU memory used during the embedding computation (in bytes).

**Processing Logic:**

1. **Tokenization and Encoding:** Uses the provided `tokenizer` to tokenize and encode the input `texts`.
2. **Model Inference:**  Passes the encoded inputs to the embedding `model` to generate raw embeddings.
3. **Mean Pooling:** Applies mean pooling to the raw embeddings to obtain a fixed-size representation for each text string.
4. **Memory Management:**
    - If a GPU is being used, it tracks and returns the maximum GPU memory consumption (`max_mem`) during the process.
    - Clears the GPU cache after computation to free up memory.
5. Returns the computed embeddings tensor and the maximum GPU memory used. 

### `get_qa_pair_embeddings(qa_pairs: list[Tuple[str, str]], model: torch.nn.Module, max_length: int, model_max_length: int) -> Tuple[np.ndarray, int]`

Generates embeddings specifically for question-answer pairs.

**Inputs:**

- `qa_pairs`:  A list of (question, answer) string tuples.
- `model`:  The loaded embedding model.
- `max_length`:  The maximum token length allowed for embedding.
- `model_max_length`: The maximum sequence length supported by the model. 

**Outputs:** 

- `qa_pair_embeddings`: A NumPy array containing the concatenated question and answer embeddings for each pair. 
- `max_mem`:  The maximum GPU memory used during the process.

**Processing Logic:**

1. **Tokenization and Truncation:** 
   - Initializes a tokenizer (using `AutoTokenizer`).
   - Truncates both the question and answer in each pair to the specified `max_length` to fit within the model's limitations.
2. **Embedding Computation:** 
   - Calls `get_embeddings` to compute embeddings for the truncated questions and answers separately. 
3. **Concatenation:** Concatenates the question and answer embeddings along the last dimension to create a single embedding for each question-answer pair.
4. **Memory Management:** Tracks and clears the GPU cache as in `get_embeddings`. 
5. Returns the combined question-answer embeddings array and the maximum GPU memory used. 

### `compute_embeddings(sharegpt_data: list[dict], config: dict) -> list[dict]`

Computes and adds embeddings to the ShareGPT data.

**Inputs:**

- `sharegpt_data`: The list of conversation dictionaries.
- `config`: The loaded configuration dictionary.

**Outputs:**

- The `sharegpt_data` list with added `qa_pair_embeddings` for each conversation.

**Processing Logic:**

1. **Model Loading and Setup:**
    - Loads the specified embedding model (from Hugging Face).
    - Sets the model to evaluation mode and moves it to the appropriate device (CPU or GPU).
2. **Batch Processing:**
    - Computes embeddings in batches to manage memory usage effectively. 
    - Dynamically adjusts batch size based on available memory.
3. **Embedding Storage:** Stores the computed `qa_pair_embeddings` within each conversation's dictionary in the `sharegpt_data` list.
4. Returns the updated `sharegpt_data`.

### `get_optimal_bins(sharegpt_data: list[dict], config: dict) -> Tuple[int, np.ndarray]`

Calculates the optimal number of bins for histogram-based filtering based on the distribution of conversation sizes. 

**Inputs:** 
- `sharegpt_data`: The list of conversation dictionaries.
- `config`: The loaded configuration dictionary.

**Outputs:** 
- `best_nbins`:  The calculated optimal number of bins.
- `final_bin_indices`: A NumPy array assigning each conversation to its corresponding bin index.

**Processing Logic:**

1. **Checks for Override:** If `force_num_bins` is set in the configuration, it uses that value and directly assigns conversations to bins.
2. **Automatic Bin Selection:** If no override, it iterates through a range of possible bin numbers, calculating the coefficient of variation (CV) for each binning scheme.
3. **CV Optimization:**  The binning scheme with the lowest CV (meaning more even distribution of data across bins) is selected. 
4. Returns the optimal number of bins and the bin assignments.

### `deduplication(sharegpt_data: list[dict], config: dict, bin_indices: np.ndarray) -> list[dict]`

Removes duplicate conversations within each bin based on cosine similarity. 

**Inputs:**

- `sharegpt_data`:  The list of conversation dictionaries.
- `config`: The loaded configuration dictionary.
- `bin_indices`: The bin assignments for each conversation.

**Outputs:**

- `deduplicated_data`: The list of conversations with duplicates removed.

**Processing Logic:**

1. **Iterate through Bins:** Iterates through each unique bin index.
2. **Deduplication within Bin:**
    - Calculates cosine similarities between the embeddings of all conversations in the current bin.
    - Identifies pairs with a similarity above the configured threshold.
    - Removes one of the conversations from each duplicate pair.
3. Returns the deduplicated dataset.

### `histogram_filter(sharegpt_data: list[dict], nbins: int, bin_indices: np.ndarray, config: dict) -> list[dict]`

Applies histogram-based filtering to normalize the distribution of conversation sizes across bins, aiming for a more balanced representation. 

**Inputs:**

- `sharegpt_data`: The list of conversation dictionaries.
- `nbins`: The number of bins.
- `bin_indices`: The bin assignments for each conversation.
- `config`: The loaded configuration dictionary.

**Outputs:**

- `filtered_data`:  The list of conversations after histogram-based filtering. 

**Processing Logic:**

1. **Bin Count Calculation:**  Determines the number of conversations in each bin.
2. **Target Count:**  Calculates a target number of conversations per bin, aiming for a more even distribution.
3. **Filtering by Bin:**
    - For bins with more conversations than the target count, it applies similarity-based filtering to reduce the number of conversations.
4. **Combine Filtered Data:**  Combines the filtered conversations from all bins into a single list.
5. Returns the filtered dataset.

### `normalize_sharegpt(sharegpt_data: list[dict], nbins: int, config: dict) -> list[dict]`

Further normalizes the data distribution to ensure that each bin contains an approximately equal number of conversations, creating a curriculum-like structure.

**Inputs:**

- `sharegpt_data`: The list of conversation dictionaries.
- `nbins`:  The number of bins.
- `config`: The loaded configuration dictionary.

**Outputs:**

- `normalized_data`: The list of conversations with an even distribution across bins.

**Processing Logic:**

1. **Bin Calculation:**  Calculates bin edges based on conversation sizes and assigns conversations to bins.
2. **Target Bin Size:** Determines the target number of conversations per bin.
3. **Bin Balancing:** 
    - Moves excess conversations from overpopulated bins to underpopulated bins.
    - Uses random sampling to select conversations to move. 
4. **Flattening:** Combines the balanced conversations from all bins into a single list.
5. Returns the normalized dataset.

### `save_chromaDB(sharegpt_data: list[dict], config: dict) -> None` 

Saves the embeddings and metadata to a ChromaDB collection, enabling efficient similarity search and retrieval.

**Inputs:**

- `sharegpt_data`: The list of conversation dictionaries.
- `config`: The loaded configuration dictionary.

**Outputs:**

- None (saves the data to a ChromaDB collection).

**Processing Logic:**

1. **ChromaDB Initialization:** Creates a ChromaDB client and collection (if it doesn't exist) using settings from the configuration.
2. **Data Preparation:**
    - Extracts question-answer pairs and embeddings from the `sharegpt_data`.
    - Creates IDs (using a hash function) for each conversation.
3. **Upsert to ChromaDB:**  Upserts (inserts or updates) the embeddings, question-answer pairs, and IDs into the ChromaDB collection. 
4. **Persist Changes:**  Persists the changes to the ChromaDB collection.

### `save_json_file(sharegpt_data: list[dict], output_file_json: str) -> None`

Saves the processed data to a JSON file.

**Inputs:**

- `sharegpt_data`: The list of conversation dictionaries.
- `output_file_json`: Path to the output JSON file.

**Outputs:**

- None (saves the data to the JSON file).

**Processing Logic:**

1. Opens the specified output file in write mode.
2. Uses `json.dump()` to write the `sharegpt_data` to the file in JSON format.

### `save_jsonl_file(sharegpt_data: list[dict], output_file_jsonl: str) -> None`

Saves the processed data to a JSON Lines (JSONL) file. 

**Inputs:**

- `sharegpt_data`:  The list of conversation dictionaries.
- `output_file_jsonl`: Path to the output JSONL file. 

**Outputs:**

- None (saves the data to the JSONL file).

**Processing Logic:**

1. Opens the specified output file in write mode.
2. Iterates through each conversation in `sharegpt_data`.
3. Writes the JSON representation of each conversation to a separate line in the file. 

### `save_sharegpt(sharegpt_data: list[dict], nbins: int, config: dict) -> None`

Orchestrates the final saving process, including splitting data into training and test sets and saving in the specified formats.

**Inputs:**

- `sharegpt_data`: The list of conversation dictionaries.
- `nbins`: The number of bins (used for stratified splitting).
- `config`: The loaded configuration dictionary.

**Outputs:**

- None (saves the data to the specified output files).

**Processing Logic:**

1. **Splitting:** 
   - Splits the data into training and test sets based on the `test_ratio` in the configuration. 
   - If normalization is enabled, it splits within each bin to ensure a balanced representation in both sets.
2. **Saving:** 
    - Calls `save_json_file` and/or `save_jsonl_file` to save the training, test, and combined data based on the output format settings in the configuration.

### `truncate_qa(question: str, answer: str, max_length: int, tokenizer: PreTrainedTokenizer) -> Tuple[str, str]`

Truncates question and answer texts to the maximum allowed length.

**Inputs:**

- `question`: The question text string.
- `answer`: The answer text string.
- `max_length`: The maximum allowed length (in tokens).
- `tokenizer`: The tokenizer to use for tokenization.

**Outputs:**

- `truncated_question`: The truncated question text.
- `truncated_answer`:  The truncated answer text.

**Processing Logic:**

1. Tokenizes the `question` and `answer` using the provided `tokenizer`.
2. Truncates the tokenized question and answer to the specified `max_length`.
3. Converts the truncated tokens back to text strings. 
4. Returns the truncated question and answer. 

###  `strip_out_rejection_phrases(sharegpt_data: list[dict], config: dict) -> list[dict]` 

Removes conversations containing predefined rejection phrases from the GPT model's responses.

**Inputs:**

- `sharegpt_data`: The list of conversation dictionaries.
- `config`:  The loaded configuration dictionary.

**Outputs:**

- The filtered `sharegpt_data` list without conversations containing rejection phrases.

**Processing Logic:**

1. Retrieves the list of rejection phrases from the `config`.
2. Iterates through each conversation:
    - Checks each GPT response for any rejection phrase.
    - If a rejection phrase is found, the entire conversation is removed.
3. Returns the filtered dataset.

### `mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor`

Performs mean pooling on the output of a language model (likely a Transformer) to obtain a fixed-size vector representation.

**Inputs:**

- `model_output`: The output tensor from the language model. This usually contains hidden states for all tokens in the input sequence.
- `attention_mask`: A tensor indicating which tokens in the input sequence are valid (1) and which are padding (0).

**Outputs:**

- `mean_pooled_embeddings`: A tensor containing the mean-pooled embeddings for each input sequence. 

**Processing Logic:**

1. Extracts the token embeddings from the `model_output`.
2. Expands the `attention_mask` to match the shape of the token embeddings.
3. Multiplies the token embeddings by the attention mask, effectively zeroing out the embeddings of padding tokens.
4. Sums the masked embeddings along the sequence length dimension.
5. Divides the sum by the number of valid tokens (determined from the attention mask) to get the mean-pooled embedding.
6. Returns the mean-pooled embeddings. 

### `check_system_prompt(conversation: list[dict], config: dict) -> list[dict]`

Ensures that each conversation has a valid and updated system prompt, incorporating relevant metadata if present.

**Inputs:**

- `conversation`:  A list of dictionaries representing the conversation.
- `config`: The loaded configuration dictionary.

**Outputs:**

- The updated conversation list, potentially with a modified or added system prompt. 

**Processing Logic:**

1. **Metadata Collection:**  Extracts any metadata from the conversation messages (keys other than `from` and `value`).
2. **System Prompt Update/Creation:**
   - If a system message already exists at the beginning of the conversation, it appends the collected metadata to the existing prompt.
   - If no system message exists, it creates one using either the default system prompt from the configuration or the collected metadata.
3. **Metadata Removal:**  Removes metadata from other messages in the conversation, ensuring that the system prompt is the sole source of metadata.
4. Returns the updated conversation. 

### `plot_histogram(sharegpt_data: list[dict], nbins: int) -> None`

Plots a histogram of the conversation size distribution to visualize how conversations are distributed across different lengths.

**Inputs:**

- `sharegpt_data`: The list of conversation dictionaries.
- `nbins`:  The number of bins to use in the histogram.

**Outputs:**

- None (prints the histogram to the console).

**Processing Logic:** 

1. **Bin Calculation:** Calculates bin edges based on the conversation sizes and the specified number of bins.
2. **Histogram Generation:** Uses the `np.histogram()` function to calculate the frequency of conversations within each bin.
3. **Output:** Prints a text-based representation of the histogram to the console, showing the size range of each bin and the number of conversations in that bin.

### `plot_normalized_histogram(normalized_data: list[dict], nbins: int) -> None`

Plots a histogram of the normalized data distribution, showing the conversation counts per bin after normalization.

**Inputs:**

- `normalized_data`: The list of conversation dictionaries after normalization.
- `nbins`: The number of bins.

**Outputs:**

- None (prints the normalized histogram to the console).

**Processing Logic:**

1. **Bin Data:** Divides the normalized data into bins based on the specified number of bins. 
2. **Output:** Prints a text-based representation of the normalized histogram, showing the number of conversations in each bin.

### `initialize_llama(config: dict) -> list[str]`

Sets up the command for interacting with the `llama.cpp` language model for system prompt improvement.

**Inputs:**

- `config`: The loaded configuration dictionary.

**Outputs:**

- `command`: A list of strings representing the base command for running `llama.cpp` with the specified settings. 

**Processing Logic:**

1. Constructs the base command based on the `llama.cpp` settings in the configuration file, including the model path, threading, context size, batch size, and other parameters.
2. Returns the constructed command list.

### `improve_system_prompt(sharegpt_data: list[dict], config: dict) -> list[dict]`

Uses the `llama.cpp` model to analyze conversations and generate potentially improved system prompts for each conversation. 

**Inputs:**

- `sharegpt_data`: The list of conversation dictionaries.
- `config`:  The loaded configuration dictionary.

**Outputs:**

- The `sharegpt_data` list with potentially improved system prompts in each conversation.

**Processing Logic:**

1. **Llama Initialization:**  Calls `initialize_llama()` to get the base `llama.cpp` command.
2. **Iterate through Conversations:**  Iterates through each conversation.
3. **Prompt Generation:**
    - Constructs a prompt for `llama.cpp` that includes:
        - Instructions to analyze the conversation and create a new system prompt.
        - The original system prompt.
        - The conversation history (human and GPT turns). 
    - Writes the prompt to a temporary file.
4. **Llama Execution:**  Executes `llama.cpp` with the generated prompt file, using additional parameters from the configuration for temperature, top-k sampling, etc.
5. **Output Processing:** 
   - Parses the output from `llama.cpp`, extracting the generated system prompt.
   - Replaces the original system prompt in the conversation with the improved one (if it's sufficiently long).
6. **Cleanup:** Deletes the temporary prompt file.
7. Returns the `sharegpt_data` with updated system prompts.

### `get_cosine_scores(qa_pair_embeddings: np.ndarray, filter_batch_size: int) -> np.ndarray` 

Computes pairwise cosine similarity scores between question-answer pair embeddings.

**Inputs:**

- `qa_pair_embeddings`:  A NumPy array containing question-answer pair embeddings.
- `filter_batch_size`: The batch size for computing cosine similarities.

**Outputs:** 

- `cosine_scores`: A NumPy array of the mean cosine similarity scores for each question-answer pair embedding against all other embeddings.

**Processing Logic:**

1. **Normalization:** Normalizes the `qa_pair_embeddings` using L2 normalization.
2. **Batch Computation:** 
   - Iterates through the embeddings in batches.
   - For each batch, calculates the cosine similarities between the batch embeddings and all embeddings in the `qa_pair_embeddings` array.
   - Computes the mean similarity score for each embedding in the batch.
3. Returns the array of mean cosine similarity scores.

### `get_filtered_indices(qa_pair_embeddings: np.ndarray, filter_ratio: float, config: dict) -> np.ndarray`

Determines the indices of conversations to keep after filtering, based on the chosen filtering method(s) and the `filter_ratio`.

**Inputs:**

- `qa_pair_embeddings`: The NumPy array of question-answer pair embeddings.
- `filter_ratio`: The target ratio of conversations to keep after filtering.
- `config`: The loaded configuration dictionary.

**Outputs:**

- A NumPy array of the indices of conversations to keep after filtering.

**Processing Logic:**

1. **Initial Indices:** Starts with all conversation indices.
2. **Filter Application:**
   - Applies the first filtering method (specified by `F1` in the configuration) to the embeddings and indices.
   - Applies the second filtering method (specified by `F2` in the configuration) to the already filtered indices and embeddings.
3. Returns the final set of filtered indices. 

### `sort_and_truncate(sharegpt_data: list[dict], config: dict) -> list[dict]`

Sorts the conversations by size (`nbytes`) and truncates the dataset to the maximum allowed size specified in the configuration.

**Inputs:**

- `sharegpt_data`: The list of conversation dictionaries.
- `config`: The loaded configuration dictionary.

**Outputs:**

- The sorted and potentially truncated `sharegpt_data` list.

**Processing Logic:**

1. **Sorting:** Sorts the conversations in ascending order based on their `nbytes`.
2. **Truncation:**  If the total size of the sorted conversations exceeds the maximum allowed size from the configuration, it truncates the dataset.
3. Returns the sorted and truncated dataset.

### `append_file_to_sharegpt(sharegpt_data: list[dict], json_file: str, config: dict) -> list[dict]`

Appends data from a JSON file to the existing `sharegpt_data`.

**Inputs:**

- `sharegpt_data`: The current list of conversation dictionaries.
- `json_file`: The path to the JSON file containing additional data. 
- `config`: The loaded configuration dictionary.

**Outputs:**

- The updated `sharegpt_data` list with the additional data appended.

**Processing Logic:**

1. **Load JSON File:**  Loads the data from the specified `json_file`.
2. **Append Data:** Extends the `sharegpt_data` list with the data loaded from the file.
3. Returns the updated `sharegpt_data`.


### `combine_sharegpt(config_file: str) -> None`

The main function that orchestrates the entire data processing pipeline.

**Inputs:**

- `config_file`: Path to the YAML configuration file.

**Outputs:** 

- None (the function saves the processed data to files).

**Processing Logic:**

1. **Load Configuration:** Calls `load_config` to load settings from the YAML configuration file.
2. **Load Data:** Calls `load_files` to load data from the input JSON and JSONL files.
3. **Preprocessing:** Calls `preprocess_data` to clean, validate, and standardize the data.
4. **Embedding Computation:**  Calls `compute_embeddings` to generate and store embeddings for each conversation.
5. **Optimal Binning:**  Calls `get_optimal_bins` to determine the best number of bins for histogram-based filtering. 
6. **Deduplication:** Calls `deduplication` to remove duplicate conversations.
7. **Histogram Filtering:** Calls `histogram_filter` to balance the data distribution across bins.
8. **Normalization:** Calls `normalize_sharegpt` to further normalize the data distribution.
9. **System Prompt Improvement:**  Calls `improve_system_prompt` (if enabled) to improve system prompts using `llama.cpp`. 
10. **Append Unmodified Files:**  Loads and appends data from any unmodified files specified in the configuration.
11. **Sorting and Truncation:**  Calls `sort_and_truncate` to sort the data by size and truncate it if necessary.
12. **Saving:**  Calls `save_sharegpt` to split the data into training and test sets and save them to the specified output files. 

### `main() -> None`

The entry point of the script. It parses the configuration file path from command-line arguments and calls the `combine_sharegpt` function. 
