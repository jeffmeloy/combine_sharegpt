import os
import re
import gc
import sys
import json
import math
import yaml
import random
import logging
from typing import Tuple

import torch
import numpy as np
from scipy.stats import entropy
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizer
from sklearn.preprocessing import normalize as sklearn_normalize

class SingleLineHandler(logging.StreamHandler):
    """custom handler to overwrite the same line"""

    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            stream.write(f"\r{msg}")
            self.flush()
        except Exception:
            self.handleError(record)


def load_config(config_path):
    """Load the configuration file"""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def update_progress(batch_idx: int, num_batches: int) -> None:
    """update the progress of the batch processing"""
    progress = (batch_idx + 1) / num_batches
    print(f"\rProcessed {batch_idx + 1}/{num_batches} batches ({progress:.1%})", end="")


def get_batch_indices(
    batch_idx: int, batch_size: int, num_pairs: int
) -> Tuple[int, int]:
    """get the start and end indices for the current batch"""
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, num_pairs)
    return start_idx, end_idx


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalize embeddings based on the minimum and maximum values."""
    min_value, max_value = np.min(embeddings), np.max(embeddings)
    scaling_factor = max(max_value - min_value, 1)
    return (embeddings - min_value) / scaling_factor


def select_top_indices(
    scores: np.ndarray, indices: np.ndarray, filter_ratio: float
) -> np.ndarray:
    """select top indices based on provided scores."""
    num_selected = int(len(indices) * filter_ratio)
    ranked_indices = np.argsort(scores)[::-1]
    selected_indices = ranked_indices[:num_selected]
    indices = np.array(indices)
    return indices[selected_indices]


def filter_by_cs(
    embeddings: np.ndarray, indices: np.ndarray, filter_ratio: float, filter_batch_size: int
) -> np.ndarray:
    """filter the question-answer pairs based on the cosine similarity, favoring diversity."""
    cosine_scores = get_cosine_scores(embeddings, filter_batch_size)
    dissimilarity_scores = (
        1 - cosine_scores[indices]
    )  # subtract from 1 to convert similarity to dissimilarity
    return select_top_indices(dissimilarity_scores, indices, filter_ratio)


def filter_by_kl_divergence(
    embeddings: np.ndarray, indices: np.ndarray, filter_ratio: float
) -> np.ndarray:
    """filter the question-answer pairs based on the KL divergence."""
    embedding_size = embeddings.shape[1] // 2
    q_emb = normalize_embeddings(embeddings[indices, :embedding_size])
    a_emb = normalize_embeddings(embeddings[indices, embedding_size:])
    kl_div = entropy(q_emb, a_emb, axis=1)
    return select_top_indices(kl_div, indices, filter_ratio)


def filter_randomly(indices: np.ndarray, filter_ratio: float) -> np.ndarray:
    """filter the question-answer pairs randomly."""
    num_pairs = len(indices)
    num_selected = int(num_pairs * filter_ratio)
    return np.random.choice(indices, num_selected, replace=False)


def filter_by_entropy(
    embeddings: np.ndarray, indices: np.ndarray, filter_ratio: float
) -> np.ndarray:
    """filter the question-answer pairs based on the entropy."""
    entropy_values = entropy(embeddings[indices], axis=1)
    return select_top_indices(entropy_values, indices, filter_ratio)


def filter_by_variance_increase(
    embeddings: np.ndarray, indices: np.ndarray, filter_ratio: float
) -> np.ndarray:
    """filter the question-answer pairs based on the variance increase."""
    mean_embedding = np.mean(embeddings[indices], axis=0)
    distances = np.linalg.norm(embeddings[indices] - mean_embedding, axis=1)
    return select_top_indices(-distances, indices, filter_ratio)


def pytorch_cosine_similarity(A, B):
    """Compute cosine similarity between A and B tensors using PyTorch"""
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
    
    A = torch.tensor(A, dtype=torch.float32, device=device)
    B = torch.tensor(B, dtype=torch.float32, device=device)
    return torch.mm(A, B.t()).cpu().numpy()


def sort_and_truncate(
    sharegpt_data: list[dict[str, any]], config: dict[str, any]
) -> list[dict[str, any]]:
    """Remove invalid sharegpt, sort data based on number of bytes, truncate and return."""
    sharegpt_data = sorted(sharegpt_data, key=lambda x: x["nbytes"])
    sharegpt_data = [
        entry
        for entry in sharegpt_data
        if entry["nbytes"] <= config["filtering"]["max_output_size"]
    ]
    return sharegpt_data


def append_file_to_sharegpt(
    sharegpt_data: list[dict[str, any]], json_file: str, config: dict[str, any]
) -> list[dict[str, any]]:
    """append sharegpt file to sharedata"""
    if os.path.exists(json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            algs_data = json.load(f)
        sharegpt_data.extend(algs_data)
    sharegpt_data = sort_and_truncate(sharegpt_data, config)
    return sharegpt_data


def save_sharegpt(sharegpt_data: list[dict[str, any]], nbins: int = 0) -> None:
    """Split test and train sets and save sharegpt_data .json and .jsonl files."""
    test_data = []
    train_data = []
    max_output_size = max(entry["nbytes"] for entry in sharegpt_data)

    if nbins > 0:
        for i in range(nbins):
            bin_data = [
                entry
                for entry in sharegpt_data
                if entry["nbytes"] >= i * (max_output_size / nbins)
            ]
            bin_data = [
                entry
                for entry in bin_data
                if entry["nbytes"] < (i + 1) * (max_output_size / nbins)
            ]
            random.shuffle(bin_data)
            test_data.extend(bin_data[: len(bin_data) // 10])
            train_data.extend(bin_data[len(bin_data) // 10 :])

        save_json_files(test_data, "test_sharegpt.json", "test_sharegpt.jsonl")
        save_json_files(train_data, "train_sharegpt.json", "train_sharegpt.jsonl")

    save_json_files(sharegpt_data, "total_sharegpt.json", "total_sharegpt.jsonl")


def similiarity_filter(
    sharegpt_data: list[dict[str, any]], filter_ratio: float
) -> list[dict[str, any]]:
    """filter the data based on the filter ratio."""
    qa_pair_embeddings = np.array(
        [entry["qa_pair_embeddings"] for entry in sharegpt_data]
    )
    filter_indices = get_filtered_indices(qa_pair_embeddings, filter_ratio)
    filtered_data = [sharegpt_data[i] for i in filter_indices]

    return filtered_data


def save_json_files(
    sharegpt_data: list[dict[str, any]],
    output_file_json: str,
    output_file_jsonl: str,
) -> None:
    """save the sharegpt data to .json and .jsonl files."""
    sorted_data = sorted(sharegpt_data, key=lambda x: x["nbytes"])
    with open(output_file_json, "w", encoding="utf-8") as f:
        json.dump(sorted_data, f, indent=4)
    with open(output_file_jsonl, "w", encoding="utf-8") as f:
        jsonl_data = [
            json.dumps(entry["conversation"], ensure_ascii=False)
            for entry in sorted_data
        ]
        f.write("\n".join(jsonl_data))


def truncate_qa(
    question: str, answer: str, max_length: int, tokenizer: PreTrainedTokenizer
) -> Tuple[str, str]:
    """truncate the question and answer based on the maximum length."""
    tokens_question = tokenizer.tokenize(question, truncation=True)
    max_q = min(max_length, len(tokens_question))
    truncated_question = tokenizer.convert_tokens_to_string(tokens_question[:max_q])
    tokens_answer = tokenizer.tokenize(answer, truncation=True)
    max_a = min(max_length, len(tokens_answer))
    truncated_answer = tokenizer.convert_tokens_to_string(tokens_answer[:max_a])
    return truncated_question, truncated_answer


def strip_out_rejection_phrases(
    sharegpt_data: list[dict[str, any]], config
) -> list[dict[str, any]]:
    """strip out rejection phrases from the conversation data"""
    rejection_phrases = config["preprocessing"]["rejection_phrases"]
    ai_pattern = re.compile("|".join(rejection_phrases))
    for entry in sharegpt_data:
        filtered_conversation = [
            item
            for item in entry["conversation"]
            if not any(
                ai_pattern.search(str(v)) for v in item.values() if isinstance(v, str)
            )
        ]
        filtered_conversation = filtered_conversation or entry["conversation"]
        filtered_conversation = [
            {**item, "value": item["value"].strip()} for item in filtered_conversation
        ]
        entry["conversation"] = filtered_conversation

    return sharegpt_data


def is_valid_conversation(
    conversation: list[dict[str, any]],
) -> Tuple[bool, list[dict[str, any]]]:
    """Check if the conversation is valid and replace 'user' with 'human' and 'assistant' with 'gpt'."""

    # Replace any "role" key with "from" key and replace any "content" key with "value" key
    for item in conversation:
        if "role" in item:
            item["from"] = item.pop("role")
        if "content" in item:
            item["value"] = item.pop("content")

    # Replace any from: "user" with from: "human" and "assistant" with "gpt"
    for item in conversation:
        if item.get("from") == "user":
            item["from"] = "human"
        elif item.get("from") == "assistant":
            item["from"] = "gpt"

    has_human = any(
        item.get("from") == "human" and item.get("value") for item in conversation
    )
    has_gpt = any(
        item.get("from") == "gpt" and item.get("value") for item in conversation
    )

    human_before_gpt = True
    last_speaker = None
    for item in conversation:
        if item.get("from") == "gpt" and last_speaker != "human":
            human_before_gpt = False
            break
        last_speaker = item.get("from")

    is_valid = human_before_gpt and has_human and has_gpt
    return is_valid, conversation


def strip_gptisms(conversation: list[dict[str, str]], config) -> list[dict[str, str]]:
    """Strip out gptisms from the conversation data."""
    for item in conversation:
        if item.get("from") == "gpt":
            for phrase in config["preprocessing"]["gpt_strip_out"]:
                item["value"] = item["value"].replace(phrase, "").strip()
    return conversation


def plot_histogram(sharegpt_data: list[dict[str, any]], nbins: int) -> None:
    """Plot the histogram of the number of bytes."""
    nbytes_array = np.array([entry["nbytes"] for entry in sharegpt_data])
    min_nbytes, max_nbytes = np.min(nbytes_array), np.max(nbytes_array)
    nbins = min(nbins, max_nbytes - min_nbytes)
    bin_edges = np.linspace(min_nbytes, max_nbytes, nbins + 1)
    histogram, _ = np.histogram(nbytes_array, bins=bin_edges)

    print("Histogram:")
    max_count = np.max(histogram)
    scale_factor = 100 / max_count
    for i in range(nbins):
        count = histogram[i]
        bar = "*" * int(count * scale_factor)
        print(f"{bin_edges[i]:.0f}-{bin_edges[i+1]:.0f}: {bar}, {count} records")


def get_cosine_scores(qa_pair_embeddings: np.ndarray, filter_batch_size) -> np.ndarray:
    """Compute the cosine similarity scores for the question-answer pair embeddings."""
    print("Calculating cosine similarity scores")

    num_qa_pairs = len(qa_pair_embeddings)
    num_batches = (num_qa_pairs + filter_batch_size - 1) // filter_batch_size
    qa_pair_embeddings = sklearn_normalize(qa_pair_embeddings, norm='l2', axis=1, copy=False)
    cosine_scores = np.zeros(num_qa_pairs)
    for batch_idx in range(num_batches):
        start_idx, end_idx = get_batch_indices(
            batch_idx, filter_batch_size, num_qa_pairs
        )
        batch_embeddings = qa_pair_embeddings[start_idx:end_idx]
        batch_similarity_matrix = pytorch_cosine_similarity(
            batch_embeddings, qa_pair_embeddings
        )
        cosine_scores[start_idx:end_idx] = np.sum(batch_similarity_matrix, axis=1)
        update_progress(batch_idx, num_batches)

    print()  # Add a newline

    return cosine_scores


def get_filtered_indices(
    qa_pair_embeddings: np.ndarray, filter_ratio: float, config: dict
) -> np.ndarray:
    """Filter the question-answer pair embeddings based on the config."""
    num_qa_pairs = len(qa_pair_embeddings)
    indices = list(range(num_qa_pairs))
    if filter_ratio >= 1:
        return indices

    filter_batch_size = config["filtering"]["filter_batch_size"]
    filters = [config["filtering"]["F1"], config["filtering"]["F2"]]
    if any(f != "none" for f in filters):
        filter_ratio = math.sqrt(filter_ratio)

    kwargs = {
        "qa_pair_embeddings": qa_pair_embeddings,
        "indices": indices,
        "filter_ratio": filter_ratio,
        "filter_batch_size": filter_batch_size
    }
    
    input_count = num_qa_pairs
    for i, filter_type in enumerate(filters, 1):
        if filter_type != "none":
            print(f"Applying {filter_type} filtering...")

            if filter_type == "cosine":
                indices = filter_by_cs(**kwargs)
            elif filter_type == "kl":
                indices = filter_by_kl_divergence(**kwargs)
            elif filter_type == "entropy":
                indices = filter_by_entropy(**kwargs)
            elif filter_type == "variance_increase":
                indices = filter_by_variance_increase(**kwargs)
            elif filter_type == "rand":
                indices = filter_randomly(**kwargs)
            else:
                print(f"Unknown filter type {filter_type}. Using previous indices.")

            output_count = len(indices)
            print(
                f"Filter #{i} - Input count: {input_count}, Output count: {output_count}"
            )
            input_count = output_count
        else:
            print(f"Filter #{i} - No filtering applied.")

    return indices


def normalize_histogram(
    sharegpt_data: list[dict[str, any]],
    nbins: int,
    bin_indices: np.ndarray,
    config: dict[str, any],
) -> list[dict[str, any]]:
    """Normalize the data count in each histogram bin."""

    norm = config["filtering"]["norm"]
    if not norm:
        return sharegpt_data

    nbytes_array = np.array([entry["nbytes"] for entry in sharegpt_data])
    min_nbytes, max_nbytes = nbytes_array.min(), nbytes_array.max()

    # Perform the optimal bin calculation with multiple passes
    bin_counts = np.bincount(bin_indices, minlength=nbins)
    avg_count = bin_counts.sum() / np.count_nonzero(bin_counts)
    min_count = max(avg_count / nbins, min_nbytes)
    bin_rank = np.argsort(bin_counts)[::-1]

    print("Input data size:", len(sharegpt_data))
    print("Number of bins:", nbins)
    print("Min nbytes:", min_nbytes)
    print("Max nbytes:", max_nbytes)
    print("Bin counts:", bin_counts)
    print("Bin rank:", bin_rank)

    normalized_data = []
    for i, bin_index in enumerate(bin_rank):
        bin_data = [
            entry
            for entry, index in zip(sharegpt_data, bin_indices)
            if index == bin_index
        ]
        if bin_counts[bin_index] > 2:
            scaled_count = min_count * max(1, nbins - i)
            max_bin_length = max(entry["nbytes"] for entry in bin_data)
            print(
                f"Bin {bin_index}: {len(bin_data)} records, scaled count: {scaled_count}, min count: {min_count}, max bin length: {max_bin_length}"
            )
            filter_ratio = scaled_count / len(bin_data)
            bin_data = similiarity_filter(bin_data, filter_ratio)
            normalized_data.extend(bin_data)

        else:  # add all bin_data records if bin_counts[bin_index] is 1 or 2
            if bin_counts[bin_index] > 0:
                normalized_data.extend(bin_data)

    return normalized_data


def get_optimal_bins(
    sharegpt_data: list[dict[str, any]], config: dict[str, any]
) -> Tuple[int, np.ndarray]:
    """Get the optimal number of bins based on the data distribution."""
    force_num_bins = config["filtering"]["force_num_bins"]
    nbytes_array = np.array([entry["nbytes"] for entry in sharegpt_data])
    min_nbytes, max_nbytes = nbytes_array.min(), nbytes_array.max()

    if isinstance(force_num_bins, int):
        nbins = force_num_bins
        bin_edges = np.linspace(min_nbytes, max_nbytes, nbins + 1)
        bin_indices = np.digitize(nbytes_array, bin_edges) - 1
        return nbins, bin_indices

    print("Obtaining optimal number of bins to normalize data distribution size...")
    best_nbins = 1
    min_cv = float("inf")
    final_bin_indices = None
    nbins_range = np.arange(2, max_nbytes + 1)
    for nbins in nbins_range:
        bin_edges = np.linspace(min_nbytes, max_nbytes, nbins + 1)
        bin_indices = np.digitize(nbytes_array, bin_edges) - 1
        bin_counts = np.bincount(bin_indices, minlength=nbins)
        avg_count = bin_counts.sum() / np.count_nonzero(bin_counts)
        min_count = max(avg_count / nbins, min_nbytes)
        bin_rank = np.argsort(bin_counts)[::-1]
        bin_deviation = np.std(bin_counts)

        for rank, _ in enumerate(bin_rank):
            scaled_count = min_count * max(1, nbins - rank)
            cv = scaled_count / bin_deviation
            if cv < min_cv:
                min_cv = cv
                best_nbins = nbins
                final_bin_indices = bin_indices

        # stop if cv = 0 because optimal or if nbins > avg_count because won't improve distribution
        if nbins > 1 and (cv == 0 or nbins > avg_count):
            break
       
    return best_nbins, final_bin_indices


def deduplication(
    sharegpt_data: list[dict[str, any]], config: dict[str, any]
) -> list[dict[str, any]]:
    """Remove duplicate question-answer pairs based on cosine similarity."""
    dedup = config["filtering"]["dedup"]
    if not dedup:
        return sharegpt_data

    similarity_threshold = config["filtering"]["similarity_threshold"]
    filter_batch_size = config["filtering"]["filter_batch_size"]

    qa_pair_embeddings = np.array(
        [entry["qa_pair_embeddings"] for entry in sharegpt_data]
    )
    qa_pair_embeddings = sklearn_normalize(qa_pair_embeddings, norm='l2', axis=1, copy=False)

    num_qa_pairs = len(qa_pair_embeddings)
    num_batches = (num_qa_pairs + filter_batch_size - 1) // filter_batch_size

    print("Removing duplicates based on cosine similarity...")
    deduplicated_mask = np.ones(num_qa_pairs, dtype=bool)

    for batch_idx in range(num_batches):
        start_idx, end_idx = get_batch_indices(
            batch_idx, filter_batch_size, num_qa_pairs
        )
        batch_embeddings = qa_pair_embeddings[start_idx:end_idx]
        cosine_similarities = pytorch_cosine_similarity(
            batch_embeddings, qa_pair_embeddings
        )
        duplicate_indices = np.where(cosine_similarities > similarity_threshold)

        filtered_indices = {
            (min(start_idx + i, j), max(start_idx + i, j))
            for i, j in zip(*duplicate_indices)
            if start_idx + i != j
        }

        seen = set()
        for i, j in filtered_indices:
            if j not in seen:
                deduplicated_mask[j] = False
                seen.add(j)
            elif i not in seen:
                deduplicated_mask[i] = False
                seen.add(i)

        sys.stdout.write(f"\rBatch idx: {batch_idx}, Num batches: {num_batches}")

    print()
    deduplicated_indices = np.where(deduplicated_mask)[0]
    deduplicated_data = [sharegpt_data[i] for i in deduplicated_indices]

    num_removed = num_qa_pairs - len(deduplicated_indices)
    print(f"Removed duplicate pairs: {num_removed}")

    return deduplicated_data


def mean_pooling(
    model_output: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Compute the mean-pooled embeddings based on the model output and attention mask."""
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    mean_pooled_embeddings = torch.sum(
        token_embeddings * input_mask_expanded, 1
    ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return mean_pooled_embeddings


def get_qa_pairs(data: list[dict[str, any]]) -> list[tuple[str, str]]:
    """Extract question-answer pairs from the data."""
    qa_pairs = []
    for entry in data:
        question_parts = []
        answer_parts = []
        question_parts = [
            item["value"]
            for item in entry["conversation"]
            if item.get("from") == "human"
        ]
        answer_parts = [
            item["value"] for item in entry["conversation"] if item.get("from") == "gpt"
        ]

        question = " [SEP] ".join(question_parts)
        answer = " [SEP] ".join(answer_parts)
        qa_pairs.append(("search_query: " + question, "search_query: " + answer))

    return qa_pairs


def get_embeddings(texts, model, tokenizer) -> Tuple[torch.Tensor, int]:
    """get the embedding for the texts using the model and tokenizer."""
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(
        model.device
    )
    outputs = model(**inputs)
    embeddings = mean_pooling(outputs, inputs["attention_mask"])

    if model.device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        max_mem = total - free
        inputs = None
        outputs = None
        del inputs, outputs
        torch.cuda.empty_cache()
    else:
        max_mem = 1

    return embeddings, max_mem


def get_qa_pair_embeddings(qa_pairs, model, max_length, model_max_length):
    """get the embeddings for the question-answer pairs using the model."""
    max_token_length = min(max_length, model_max_length)
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-uncased", model_max_length=max_token_length
    )
    questions, answers = zip(
        *[truncate_qa(q, a, max_token_length, tokenizer) for q, a in qa_pairs]
    )

    q_emb, max_mem_q = get_embeddings(questions, model, tokenizer)
    a_emb, max_mem_a = get_embeddings(answers, model, tokenizer)

    max_mem = max(max_mem_q, max_mem_a)
    free, total = torch.cuda.mem_get_info()
    max_mem = max(max_mem, total - free)

    qa_pair_embeddings = torch.cat((q_emb, a_emb), dim=1).cpu().numpy()

    q_emb = None
    a_emb = None
    tokenizer = None
    del q_emb, a_emb, tokenizer

    if model.device.type == "cuda":
        torch.cuda.empty_cache()

    return qa_pair_embeddings, max_mem


@torch.no_grad()
def compute_embeddings(sharegpt_data, config):
    """Compute the embeddings for the question-answer pairs in the sharegpt data."""

    min_nbytes = min(entry["nbytes"] for entry in sharegpt_data)
    max_nbytes = max(entry["nbytes"] for entry in sharegpt_data)
    total_entries = len(sharegpt_data)
    print(f"Minimum nbytes: {min_nbytes}")
    print(f"Maximum nbytes: {max_nbytes}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_max_length = config["embedding"]["max_length"]
    embedding_model_path = config["embedding"]["embedding_model_path"]
    trust_remote_code = config["embedding"]["trust_remote_code"]
    safe_serialization = config["embedding"]["safe_serialization"]
    rotary_scaling_factor = config["embedding"]["rotary_scaling_factor"]
    local_files_only = config["embedding"]["local_files_only"]
    initial_batch_size = config["embedding"]["initial_batch_size"]
    memory_fraction = config["embedding"]["memory_fraction"]

    try:
        model = AutoModel.from_pretrained(
            embedding_model_path,
            trust_remote_code=trust_remote_code,
            safe_serialization=safe_serialization,
            rotary_scaling_factor=rotary_scaling_factor,
            device_map=device,
            local_files_only=local_files_only,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return sharegpt_data

    model.eval()
    model.to(device)
    print(f"Using device: {device}")

    # Constants for batching
    free, total = torch.cuda.mem_get_info()
    batch_max_mem = total * memory_fraction
    max_mem = batch_max_mem
    current_batch_size = initial_batch_size

    batch_start_idx = 0
    processed_entries = 0
    for i, entry in enumerate(sharegpt_data):
        batch_max_length = entry["nbytes"]
        scale = batch_max_mem / max_mem
        batch_size = min(max(int(current_batch_size * scale), 1), current_batch_size)

        # Ensure batch_size does not exceed the remaining entries
        if batch_start_idx + batch_size > total_entries:
            batch_size = total_entries - batch_start_idx

        # Process batch if batch size is reached or it's the last entry
        if (i - batch_start_idx) + 1 >= batch_size or i == total_entries - 1:
            end_idx = batch_start_idx + batch_size
            qa_pairs = get_qa_pairs(sharegpt_data[batch_start_idx:end_idx])
            batch_embeddings, max_mem = get_qa_pair_embeddings(
                qa_pairs, model, batch_max_length, model_max_length
            )

            for j, batch_entry in enumerate(sharegpt_data[batch_start_idx:end_idx]):
                batch_entry["qa_pair_embeddings"] = batch_embeddings[j]

            processed_entries += len(sharegpt_data[batch_start_idx:end_idx])
            progress = processed_entries / total_entries * 100
            free, total = torch.cuda.mem_get_info()
            max_mem = max(max_mem, total - free)

            sys.stdout.write(
                f"\rComputing Embeddings: {processed_entries}/{total_entries} entries ({progress:.2f}%), "
                f"batch size: {batch_size}, batch max length: {batch_max_length}, max mem: {max_mem:.2f} Bytes "
            )

            batch_start_idx = end_idx
            current_batch_size = batch_size

            qa_pairs = None
            batch_embeddings = None
            del qa_pairs, batch_embeddings
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print("\nEmbedding computation completed.")
    model = None
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return sharegpt_data


def preprocess_data(
    sharegpt_data: list[dict[str, any]], config: dict[str, any]
) -> list[dict[str, any]]:
    """Preprocess the sharegpt data."""

    # sort and truncate sharegpt_data
    sharegpt_data = sort_and_truncate(sharegpt_data, config)

    # remove GPT-isms from the conversation
    processed_data = []
    for entry in sharegpt_data:
        modified_conversation = strip_gptisms(entry["conversation"], config)
        processed_data.append(entry)
    sharegpt_data = processed_data

    # validate conversations and replace "user" with "human" and "assistant" with "gpt"
    processed_data = []
    for entry in sharegpt_data:
        is_valid, modified_conversation = is_valid_conversation(entry["conversation"])
        if is_valid:
            entry["conversation"] = modified_conversation
            processed_data.append(entry)

    # remove conversations with rejection phrases
    processed_data = strip_out_rejection_phrases(processed_data, config)

    # remove exact duplicate conversations
    unique_conversations = set()
    unique_processed_data = []
    for entry in processed_data:
        conversation_tuple = tuple(
            tuple(item.items()) for item in entry["conversation"]
        )
        if conversation_tuple not in unique_conversations:
            unique_conversations.add(conversation_tuple)
            unique_processed_data.append(entry)

    # remove conversations that don't have both human and gpt responses and values
    sharegpt_data = [
        entry
        for entry in unique_processed_data
        if any(item.get("from") == "human" for item in entry["conversation"])
        and any(item.get("from") == "gpt" for item in entry["conversation"])
    ]

    return sharegpt_data


def check_system_prompt(conversation, config):
    """If no system prompt add the one from the config."""
    processed_conversation = []
    if not any(item.get("from") == "system" for item in conversation):
        processed_conversation.append({"from": "system", "value": ""})

    system_prompt = config["preprocessing"]["system_prompt"]
    processed_conversation = [
        {**item, "value": system_prompt}
        if item.get("from") == "system" and item.get("value") == ""
        else item
        for item in conversation
    ]
    return processed_conversation


def load_json_files(config) -> list[dict[str, any]]:
    """Load the json files in the directory and return the combined data."""

    directory = config["input"]["directory"]
    json_files = [
        file
        for file in os.listdir(directory)
        if file.endswith(".json")
        and file
        not in [
            "train_sharegpt.json",
            "test_sharegpt.json",
            "total_sharegpt.json",
            "algs.json",
        ]
    ]

    combined_data = []
    for file in json_files:
        print(f"Loading file: {file}")
        with open(os.path.join(directory, file), "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error loading file: {file}, JSON decoding error: {e}")
                continue

            # replace the "conversations" key with "conversation" key 
            for entry in data:
                if "conversations" in entry:
                    entry["conversation"] = entry.pop("conversations")

            for entry in data:
                entry["conversation"] = check_system_prompt(
                    entry.get("conversation", []), config
                )
                entry["nbytes"] = sum(
                    len(json.dumps(msg, ensure_ascii=False).encode("utf-8"))
                    for msg in entry["conversation"]
                )
            combined_data.extend(data)

    return combined_data


def combine_sharegpt(config_file: str) -> None:
    """
    Combine the sharegpt data from multiple files, preprocess, compute embeddings, remove duplicates, normalize, and save the data.
    sharegpt format:
    [
        {
            "conversation": [
                {
                    "from": "system",
                    "value": <system_value>
                },
                {
                    "from": "human",
                    "value": <human_value>
                },
                {
                    "from": "gpt",
                    "value": <gpt_value>
                },
                ...
            ],
            "source": <source>,
            "nbytes": <nbytes of conversation>
        },
        ...
    ]
    """
    # Check if the config file exists
    if not os.path.exists(config_file):
        print(f"Error: Configuration file '{config_file}' not found.")
        sys.exit(1)

    # Load the configuration
    config = load_config(config_file)

    # read in and combine the *.sharegpt.json files in the directory
    sharegpt_data = load_json_files(config)
    total_input_records = len(sharegpt_data)
    print(f"Total input records: {total_input_records}")

    # sort from smallest to largest and remove gpt-isms and identical conversations
    sharegpt_data = preprocess_data(sharegpt_data, config)
    total_output_records = len(sharegpt_data)
    print(f"Total output records after preprocessing: {total_output_records}")

    # compute embeddings or skip if already computed
    if not all("qa_pair_embeddings" in entry for entry in sharegpt_data):
        sharegpt_data = compute_embeddings(sharegpt_data, config)

    # get the optimal number of bins based on the data
    nbins, bin_indicies = get_optimal_bins(sharegpt_data, config)
    print(f"Optimal number of bins: {nbins}")

    # deduplicate based on cosine similarity
    sharegpt_data = deduplication(sharegpt_data, config)

    # normalize the histogram based on the optimal number of bins
    sharegpt_data = normalize_histogram(sharegpt_data, nbins, bin_indicies, config)
    total_output_records = len(sharegpt_data)
    print(f"Total output records after normalization: {total_output_records}")

    # delete qa_pair_embeddings to save space
    sharegpt_data = [
        {k: v for k, v in entry.items() if k != "qa_pair_embeddings"}
        for entry in sharegpt_data
    ]

    # code_data = "algs.json"
    # sharegpt_data = append_file_to_sharegpt(sharegpt_data, code_data, config)

    total_output_records = len(sharegpt_data)
    print(f"Total input records: {total_input_records}")
    print(f"Total output records: {total_output_records}")
    plot_histogram(sharegpt_data, nbins)

    save_sharegpt(sharegpt_data, nbins)


def main() -> None:
    """Get configuration file and pass to combine_sharegpt."""
    config_file = (
        sys.argv[1]
        if len(sys.argv) > 1 and os.path.exists(sys.argv[1])
        else "combine_sharegpt_config.yaml"
    )
    combine_sharegpt(config_file)


if __name__ == "__main__":
    main()
