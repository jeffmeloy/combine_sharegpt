import os
import re
import gc
import sys
import json
import math
import yaml
import random
import logging
import hashlib
import jsonlines
import subprocess
from typing import Tuple
from collections import deque

import torch
import chromadb
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


def load_config(config_path: str) -> dict:
    """Load the YAML configuration file from the given path."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def update_progress(batch_idx: int, num_batches: int) -> None:
    """Update and display the progress of the batch processing."""
    progress = (batch_idx + 1) / num_batches
    print(f"\rProcessed {batch_idx + 1}/{num_batches} batches ({progress:.1%})", end="")


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Normalize the embeddings based on their minimum and maximum values."""
    min_value, max_value = np.min(embeddings), np.max(embeddings)
    scaling_factor = max(max_value - min_value, 1)
    return (embeddings - min_value) / scaling_factor


def select_top_indices(
    scores: np.ndarray, indices: np.ndarray, filter_ratio: float
) -> np.ndarray:
    """Select the top indices based on the provided scores and filter ratio."""
    num_selected = int(len(indices) * filter_ratio)
    ranked_indices = np.argsort(scores)[::-1]
    selected_indices = ranked_indices[:num_selected]
    indices = np.array(indices)
    return indices[selected_indices]


def filter_by_cs(
    embeddings: np.ndarray,
    indices: np.ndarray,
    filter_ratio: float,
    filter_batch_size: int,
) -> np.ndarray:
    """Filter question-answer pairs based on cosine similarity, favoring diversity."""
    cosine_scores = get_cosine_scores(embeddings, filter_batch_size)
    dissimilarity_scores = (
        1 - cosine_scores[indices]
    )  # subtract from 1 to convert similarity to dissimilarity
    return select_top_indices(dissimilarity_scores, indices, filter_ratio)


def get_cosine_scores(
    qa_pair_embeddings: np.ndarray, filter_batch_size: int
) -> np.ndarray:
    """Compute cosine similarity scores for question-answer pairs."""
    print("Calculating cosine similarity scores")
    num_qa_pairs = len(qa_pair_embeddings)
    num_batches = (num_qa_pairs + filter_batch_size - 1) // filter_batch_size
    qa_pair_embeddings = sklearn_normalize(
        qa_pair_embeddings, norm="l2", axis=1, copy=False
    )
    cosine_scores = np.zeros(num_qa_pairs)
    for batch_idx in range(num_batches):
        start_idx = batch_idx * filter_batch_size
        end_idx = min(start_idx + filter_batch_size, num_qa_pairs)
        batch_embeddings = qa_pair_embeddings[start_idx:end_idx]
        batch_similarity_matrix = pytorch_cosine_similarity(
            batch_embeddings, qa_pair_embeddings
        )
        cosine_scores[start_idx:end_idx] = np.mean(batch_similarity_matrix, axis=1)
        update_progress(batch_idx, num_batches)
    print()  # Add a newline
    return cosine_scores


def filter_by_kl_divergence(
    embeddings: np.ndarray, indices: np.ndarray, filter_ratio: float
) -> np.ndarray:
    """Filter question-answer pairs based on KL divergence."""
    embedding_size = embeddings.shape[1] // 2
    q_emb = normalize_embeddings(embeddings[indices, :embedding_size])
    a_emb = normalize_embeddings(embeddings[indices, embedding_size:])
    kl_div = entropy(q_emb, a_emb, axis=1)
    return select_top_indices(kl_div, indices, filter_ratio)


def filter_by_ner(
    embeddings: np.ndarray, indices: np.ndarray, filter_ratio: float
) -> np.ndarray:
    """Filter by normalized effective rank."""
    ner_scores = get_normalized_effective_rank(embeddings)
    new_scores = ner_scores[indices]
    return select_top_indices(new_scores, indices, filter_ratio)


def get_normalized_effective_rank(qa_pair_embeddings: np.ndarray) -> np.ndarray:
    """Calculate the normalized effective rank of each embedding in the given batch using NumPy."""
    num_qa_pairs = len(qa_pair_embeddings)
    ner = np.zeros(num_qa_pairs)  # This stores the NER values
    for idx, emb in enumerate(qa_pair_embeddings):
        # Since embedding is 1D use this as proxy for SVD of the embedding
        s = np.abs(emb).flatten()

        # Filter out near-zero singular values
        s = s[s > 1e-12]
        if s.size == 0:
            ner[idx] = 0.0
            continue

        # Normalize singular values and calculate entropy
        s /= s.sum()
        log_s = np.log2(s)
        entropy = -np.dot(s, log_s)

        # Compute the normalized effective rank
        max_entropy = np.log2(float(s.size))
        ner[idx] = entropy / max_entropy if max_entropy > 0 else 0.0

    return ner


def filter_randomly(indices: np.ndarray, filter_ratio: float) -> np.ndarray:
    """Randomly filter the question-answer pairs based on the filter ratio."""
    num_pairs = len(indices)
    num_selected = int(num_pairs * filter_ratio)
    return np.random.choice(indices, num_selected, replace=False)


def filter_by_entropy(
    embeddings: np.ndarray, indices: np.ndarray, filter_ratio: float
) -> np.ndarray:
    """Filter question-answer pairs based on their entropy values."""
    entropy_values = entropy(embeddings[indices], axis=1)
    return select_top_indices(entropy_values, indices, filter_ratio)


def filter_by_variance_increase(
    embeddings: np.ndarray, indices: np.ndarray, filter_ratio: float
) -> np.ndarray:
    """Filter question-answer pairs based on variance increase from the mean."""
    mean_embedding = np.mean(embeddings[indices], axis=0)
    distances = np.linalg.norm(embeddings[indices] - mean_embedding, axis=1)
    return select_top_indices(distances, indices, filter_ratio)


def pytorch_cosine_similarity(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between two matrices A and B using PyTorch."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    A = torch.tensor(A, dtype=torch.float16, device=device)
    B = torch.tensor(B, dtype=torch.float16, device=device)
    return torch.mm(A, B.t()).cpu().numpy()


def similiarity_filter(
    sharegpt_data: list[dict], filter_ratio: float, config: dict
) -> list[dict]:
    """Filter the dataset based on similarity and filter ratio."""
    qa_pair_embeddings = np.array(
        [entry["qa_pair_embeddings"] for entry in sharegpt_data]
    )
    filter_indices = get_filtered_indices(qa_pair_embeddings, filter_ratio, config)
    filtered_data = [sharegpt_data[i] for i in filter_indices]
    return filtered_data


def sort_and_truncate(sharegpt_data: list[dict], config: dict) -> list[dict]:
    """Sort the data by size and truncate based on the maximum allowed size."""
    for entry in sharegpt_data:
        entry["nbytes"] = sum(
            len(json.dumps(msg, ensure_ascii=False).encode("utf-8"))
            for msg in entry["conversation"]
            if msg.get("from") != "system"
        )

    sharegpt_data = sorted(sharegpt_data, key=lambda x: x["nbytes"])
    sharegpt_data = [
        entry
        for entry in sharegpt_data
        if entry["nbytes"] <= config["filtering"]["max_output_size"]
    ]
    return sharegpt_data


def append_file_to_sharegpt(
    sharegpt_data: list[dict], json_file: str, config: dict
) -> list[dict]:
    """Append data from a JSON file to the existing sharegpt data."""
    if os.path.exists(json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            algs_data = json.load(f)
        sharegpt_data.extend(algs_data)
    return sharegpt_data


def normalize_sharegpt(
    sharegpt_data: list[dict], nbins: int, config: dict
) -> list[dict]:
    """
    1 - compute len(sharegpt_data) / nbins round up. this is size of each bin
    2 - create normalized_data that is target_bin_size * nbins (ish) long
    3 - from first to last bin
        if bin count > size of each bin then randomly pull out data from sharegpt_data into normalize_data, and put the rest in excess_data
        if bin count < size of each bin then fill the bins in normalize_data with all of the elements in the corresponding bin from sharegpt_data with random excess_data to fill out the target_bin_size
    """
    if nbins == 0 or not config["control"]["normalize"]:
        return sharegpt_data

    print("Normalizing sharegpt data...")
    target_bin_size = math.ceil(len(sharegpt_data) / nbins)

    # obtain initial bin data
    nbytes_array = np.array([entry["nbytes"] for entry in sharegpt_data])
    min_nbytes, max_nbytes = np.min(nbytes_array), np.max(nbytes_array)
    nbins = min(nbins, max_nbytes - min_nbytes)
    bin_edges = np.linspace(min_nbytes, max_nbytes, nbins + 1)
    bin_indices = np.digitize(nbytes_array, bin_edges) - 1
    normalized_data = []
    excess_data = deque()

    # Process bins from first to last (smallest to largest)
    for bin_index in range(nbins):
        bin_data = [
            entry for entry, idx in zip(sharegpt_data, bin_indices) if idx == bin_index
        ]
        print(f"Processing Bin {bin_index + 1}: {len(bin_data)} records")
        if len(bin_data) > target_bin_size:
            # Randomly pull out data from sharegpt_data into normalize_data
            random.shuffle(bin_data)
            normalized_data.extend(bin_data[:target_bin_size])
            excess_data.extend(bin_data[target_bin_size:])
            print(
                f"  Added {target_bin_size} to normalized_data, {len(bin_data) - target_bin_size} to excess_data"
            )
        else:
            # Fill the bins in normalized_data with all elements from this bin
            normalized_data.extend(bin_data)
            remaining_space = target_bin_size - len(bin_data)

            # Fill out the target_bin_size with random excess_data
            if remaining_space > 0 and excess_data:
                additional_data = random.sample(
                    list(excess_data), min(remaining_space, len(excess_data))
                )
                normalized_data.extend(additional_data)
                excess_data = deque(
                    entry for entry in excess_data if entry not in additional_data
                )
                print(
                    f"  Added {len(bin_data)} original records and {len(additional_data)} from excess_data"
                )
            else:
                print(f"  Added {len(bin_data)} original records")

    # Shuffle each bin in normalized_data
    print("Shuffling bins...")
    shuffled_data = []
    for i in range(0, len(normalized_data), target_bin_size):
        bin_end = min(i + target_bin_size, len(normalized_data))
        bin_data = normalized_data[i:bin_end]
        random.shuffle(bin_data)
        shuffled_data.extend(bin_data)
        print(f"Shuffled bin {i//target_bin_size + 1}: {len(bin_data)} records")
    normalized_data = shuffled_data

    # Print statistics
    print(f"\nTotal records after shuffling: {len(normalized_data)}")
    for i in range(0, len(normalized_data), target_bin_size):
        bin_data = normalized_data[i : i + target_bin_size]
        if bin_data:
            min_nbytes = min(entry["nbytes"] for entry in bin_data)
            max_nbytes = max(entry["nbytes"] for entry in bin_data)
            print(
                f"Shuffled Bin {i//target_bin_size + 1}: {len(bin_data)} records (nbytes range: {min_nbytes}-{max_nbytes})"
            )
    return normalized_data


def save_sharegpt(sharegpt_data: list[dict], nbins: int, config: dict) -> None:
    """Split data into training and test sets and save them in the specified format."""
    normalize = config["control"]["normalize"]
    test_ratio = config["output"]["test_ratio"]
    output_directory = config["output"]["directory"]
    train_output_file = config["output"]["train_output_file"]
    test_output_file = config["output"]["test_output_file"]
    total_output_file = config["output"]["total_output_file"]
    save_json = config["output"]["save_json"]
    save_jsonl = config["output"]["save_jsonl"]
    test_data = []
    train_data = []

    if normalize:
        # create stratified nbins segments since normalized data arranged in a curriculum order
        segment_size = len(sharegpt_data) // nbins
        for i in range(nbins):
            start_idx = i * segment_size
            end_idx = start_idx + segment_size if i < nbins - 1 else len(sharegpt_data)
            segment_data = sharegpt_data[start_idx:end_idx]
            random.shuffle(segment_data)
            split_index = int(len(segment_data) * test_ratio)
            test_data.extend(segment_data[:split_index])
            train_data.extend(segment_data[split_index:])
    else:
        # For non-normalized data, we use nbytes-based binning
        max_output_size = max(entry["nbytes"] for entry in sharegpt_data)
        bin_edges = np.linspace(0, max_output_size, nbins + 1)
        for i in range(nbins):
            bin_data = [
                entry
                for entry in sharegpt_data
                if bin_edges[i] <= entry["nbytes"] < bin_edges[i + 1]
            ]
            random.shuffle(bin_data)
            split_index = int(len(bin_data) * test_ratio)
            test_data.extend(bin_data[:split_index])
            train_data.extend(bin_data[split_index:])

    if save_json:  # Save the split data
        save_json_file(
            train_data, os.path.join(output_directory, f"{train_output_file}.json")
        )
        save_json_file(
            test_data, os.path.join(output_directory, f"{test_output_file}.json")
        )
        save_json_file(
            sharegpt_data, os.path.join(output_directory, f"{total_output_file}.json")
        )
    if save_jsonl:
        save_jsonl_file(
            train_data, os.path.join(output_directory, f"{train_output_file}.jsonl")
        )
        save_jsonl_file(
            test_data, os.path.join(output_directory, f"{test_output_file}.jsonl")
        )
        save_jsonl_file(
            sharegpt_data, os.path.join(output_directory, f"{total_output_file}.jsonl")
        )

    # Print some statistics
    print(f"Total data size: {len(sharegpt_data)}")
    print(f"Train data size: {len(train_data)}")
    print(f"Test data size: {len(test_data)}")


def save_json_file(sharegpt_data: list[dict], output_file_json: str) -> None:
    """Save the sharegpt data to a JSON file."""
    with open(output_file_json, "w", encoding="utf-8") as f:
        json.dump(sharegpt_data, f, indent=4, ensure_ascii=False)


def save_jsonl_file(sharegpt_data: list[dict], output_file_jsonl: str) -> None:
    """Save the sharegpt data to a JSONL file."""
    with open(output_file_jsonl, "w", encoding="utf-8") as f:
        for entry in sharegpt_data:
            f.write(json.dumps(entry["conversation"], ensure_ascii=False) + "\n")


def truncate_qa(
    question: str, answer: str, max_length: int, tokenizer: PreTrainedTokenizer
) -> Tuple[str, str]:
    """Truncate the question and answer texts to the maximum allowed length."""
    tokens_question = tokenizer.tokenize(question, truncation=True)
    max_q = min(max_length, len(tokens_question))
    truncated_question = tokenizer.convert_tokens_to_string(tokens_question[:max_q])
    tokens_answer = tokenizer.tokenize(answer, truncation=True)
    max_a = min(max_length, len(tokens_answer))
    truncated_answer = tokenizer.convert_tokens_to_string(tokens_answer[:max_a])
    return truncated_question, truncated_answer


def strip_out_rejection_phrases(sharegpt_data: list[dict], config: dict) -> list[dict]:
    """Strip out predefined rejection phrases from the dataset."""
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
    conversation: list[dict], config: dict
) -> Tuple[bool, list[dict]]:
    """Check if the conversation is valid and adjust the roles accordingly."""
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

    # Apply check_system_prompt
    conversation = check_system_prompt(conversation, config)

    has_human = any(
        item.get("from") == "human" and item.get("value") for item in conversation
    )
    has_gpt = any(
        item.get("from") == "gpt" and item.get("value") for item in conversation
    )

    # if there are repeated human or gpt responses, combine them into one
    i = 1
    if has_human and has_gpt:
        while i < len(conversation):
            if (
                conversation[i].get("from") == "human"
                and conversation[i - 1].get("from") == "human"
            ):
                conversation[i - 1]["value"] += " " + conversation[i]["value"]
                conversation.pop(i)
            elif (
                conversation[i].get("from") == "gpt"
                and conversation[i - 1].get("from") == "gpt"
            ):
                conversation[i - 1]["value"] += " " + conversation[i]["value"]
                conversation.pop(i)
            else:
                i += 1

    # ensure conversation has proper order of human and gpt responses
    human_before_gpt = True
    last_speaker = None
    for item in conversation:
        if item.get("from") == "gpt" and last_speaker != "human":
            human_before_gpt = False
            break
        last_speaker = item.get("from")
    if last_speaker == "human":
        conversation = conversation[:-1]

    is_valid = human_before_gpt and has_human and has_gpt
    return is_valid, conversation


def strip_gptisms(conversation: list[dict], config: dict) -> list[dict]:
    """Remove predefined GPT-isms from the conversation data."""
    for item in conversation:
        if item.get("from") == "gpt":
            for phrase in config["preprocessing"]["gpt_strip_out"]:
                item["value"] = item["value"].replace(phrase, "").strip()
            # remove trailing Remember sentences.
            phrases = [
                "Remember, it's important",
                "Remember, it's always important",
                "Remember, it’s important",
                "Remember, it’s always important",
            ]
            for phrase in phrases:
                if phrase in item["value"]:
                    item["value"] = item["value"].split(phrase)[0].strip()

    return conversation


def plot_histogram(sharegpt_data: list[dict], nbins: int) -> None:
    """Plot a histogram of the data distribution by size."""
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


def get_filtered_indices(
    qa_pair_embeddings: np.ndarray, filter_ratio: float, config: dict
) -> np.ndarray:
    """Get the filtered indices based on the selected filtering methods."""
    num_qa_pairs = len(qa_pair_embeddings)
    indices = list(range(num_qa_pairs))
    if filter_ratio >= 1:
        return indices

    filter_batch_size = config["filtering"]["filter_batch_size"]
    filters = [config["filtering"]["F1"], config["filtering"]["F2"]]
    if any(f != "none" for f in filters):
        filter_ratio = math.sqrt(filter_ratio)

    input_count = num_qa_pairs
    for i, filter_type in enumerate(filters, 1):
        if filter_type != "none":
            print(f"Applying {filter_type} filtering...")

            if filter_type == "cosine":
                indices = filter_by_cs(
                    qa_pair_embeddings, indices, filter_ratio, filter_batch_size
                )
            elif filter_type == "kl":
                indices = filter_by_kl_divergence(
                    qa_pair_embeddings, indices, filter_ratio
                )
            elif filter_type == "ner":
                indices = filter_by_ner(qa_pair_embeddings, indices, filter_ratio)
            elif filter_type == "entropy":
                indices = filter_by_entropy(qa_pair_embeddings, indices, filter_ratio)
            elif filter_type == "variance_increase":
                indices = filter_by_variance_increase(
                    qa_pair_embeddings, indices, filter_ratio
                )
            elif filter_type == "rand":
                indices = filter_randomly(qa_pair_embeddings, indices, filter_ratio)
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


def histogram_filter(
    sharegpt_data: list[dict], nbins: int, bin_indices: np.ndarray, config: dict
) -> list[dict]:
    """Apply histogram-based filtering to normalize the dataset distribution."""

    filter = config["control"]["filter"]
    boost = config["filtering"]["boost"]
    if not filter:
        return sharegpt_data

    # Perform the optimal bin calculation with multiple passes
    nbytes_array = np.array([entry["nbytes"] for entry in sharegpt_data])
    min_nbytes, max_nbytes = nbytes_array.min(), nbytes_array.max()
    bin_counts = np.bincount(bin_indices, minlength=nbins)
    avg_count = bin_counts.sum() / np.count_nonzero(bin_counts)
    # what does min_nbytes have to do with min_count
    # should just avg_count / nbins * (1.0 - boost)??
    min_count = max(avg_count / nbins * (1.0 - boost), min_nbytes)

    bin_rank = np.argsort(bin_counts)[::-1]

    print("Input data size:", len(sharegpt_data))
    print("Number of bins:", nbins)
    print("Min nbytes:", min_nbytes)
    print("Max nbytes:", max_nbytes)
    print("Bin counts:", bin_counts)
    print("Bin rank:", bin_rank)

    filtered_data = []
    for i, bin_index in enumerate(bin_rank):
        try:
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
                bin_data = similiarity_filter(bin_data, filter_ratio, config)
                filtered_data.extend(bin_data)

            elif bin_counts[bin_index] > 0:
                filtered_data.extend(bin_data)
        except Exception as e:
            print(f"Skipping bin index: {bin_index}. Error: {str(e)}")

    return filtered_data


def get_optimal_bins(sharegpt_data: list[dict], config: dict) -> Tuple[int, np.ndarray]:
    """Calculate the optimal number of bins based on the data distribution."""
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
    sharegpt_data: list[dict], bin_indices: np.ndarray, config: dict
) -> list[dict]:
    """Remove duplicate question-answer pairs based on cosine similarity, using binning."""
    dedup = config["control"]["dedup"]
    if not dedup:
        return sharegpt_data

    similarity_threshold = config["filtering"]["similarity_threshold"]
    filter_batch_size = config["filtering"]["filter_batch_size"]
    unique_bins = np.unique(bin_indices)
    all_deduplicated_indices = []
    total_removed = 0
    total_bins = len(unique_bins)

    print(
        f"Removing duplicates based on cosine similarity, processing {total_bins} bins..."
    )
    for bin_idx in unique_bins:
        bin_data_indices = np.where(bin_indices == bin_idx)[0]
        if len(bin_data_indices) == 1:
            all_deduplicated_indices.extend(bin_data_indices)
            continue

        bin_data = [sharegpt_data[i] for i in bin_data_indices]
        qa_pair_embeddings = np.array(
            [entry["qa_pair_embeddings"] for entry in bin_data]
        )
        qa_pair_embeddings = sklearn_normalize(
            qa_pair_embeddings, norm="l2", axis=1, copy=False
        )

        num_qa_pairs = len(qa_pair_embeddings)
        num_batches = (num_qa_pairs + filter_batch_size - 1) // filter_batch_size
        deduplicated_mask = np.ones(num_qa_pairs, dtype=bool)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * filter_batch_size
            end_idx = min(start_idx + filter_batch_size, num_qa_pairs)
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

        deduplicated_indices = np.where(deduplicated_mask)[0]
        all_deduplicated_indices.extend(bin_data_indices[deduplicated_indices])
        num_removed = num_qa_pairs - len(deduplicated_indices)
        total_removed += num_removed
        sys.stdout.write(
            f"\rProcessed bin {bin_idx + 1}/{total_bins}, removed {num_removed} duplicates"
        )

    print(f"\nTotal removed duplicate pairs: {total_removed}")
    deduplicated_data = [sharegpt_data[index] for index in all_deduplicated_indices]

    return deduplicated_data


def mean_pooling(
    model_output: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Perform mean pooling on model outputs using the attention mask."""
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    mean_pooled_embeddings = torch.sum(
        token_embeddings * input_mask_expanded, 1
    ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return mean_pooled_embeddings


def get_qa_pairs(data: list[dict]) -> list[Tuple[str, str]]:
    """Extract and return question-answer pairs from the dataset."""
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


def get_embeddings(
    texts: list[str], model: torch.nn.Module, tokenizer: PreTrainedTokenizer
) -> Tuple[torch.Tensor, int]:
    """Compute embeddings for the given texts using the specified model and tokenizer."""
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


def get_qa_pair_embeddings(
    qa_pairs: list[Tuple[str, str]],
    model: torch.nn.Module,
    max_length: int,
    model_max_length: int,
) -> Tuple[np.ndarray, int]:
    """Compute embeddings for the question-answer pairs."""

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
def compute_embeddings(sharegpt_data: list[dict], config: dict) -> list[dict]:
    """Compute embeddings for the question-answer pairs in the sharegpt data."""

    if config["control"]["dedup"] is False and config["control"]["filter"] is False:
        return sharegpt_data

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


def check_system_prompt(conversation: list[dict], config: dict) -> list[dict]:
    """Ensure each conversation has a valid and updated system prompt."""
    system_prompt = config["preprocessing"]["system_prompt"]

    # Collect unique metadata
    metadata = {}
    gpt_metadata = {}

    for message in conversation:
        if message.get("from") not in ["system", "human", "gpt"]:
            for key, value in message.items():
                if key not in ["from", "value"]:
                    if isinstance(value, bool):
                        metadata[key] = "true" if value else "false"
                    else:
                        metadata[key] = str(value)
        elif message.get("from") in ["human", "gpt"]:
            for key, value in message.items():
                if key not in ["from", "value"]:
                    if isinstance(value, bool):
                        gpt_metadata[key] = "true" if value else "false"
                    else:
                        gpt_metadata[key] = str(value)

    metadata.update(gpt_metadata)
    metadata_prompt = ", ".join(
        f"{key}: {value}" for key, value in metadata.items() if value != "None"
    )

    # Create or update system message
    if conversation and conversation[0].get("from") == "system":
        existing_prompt = conversation[0].get("value", "").strip()
        if existing_prompt:
            new_prompt = f"{existing_prompt}, {metadata_prompt}".strip(", ")
        else:
            new_prompt = metadata_prompt or system_prompt
        conversation[0]["value"] = new_prompt or system_prompt
    else:
        new_prompt = metadata_prompt or system_prompt
        conversation.insert(0, {"from": "system", "value": new_prompt})

    # Remove metadata from other messages
    new_conversation = []
    for message in conversation:
        new_message = {"from": message["from"], "value": message["value"]}
        new_conversation.append(new_message)

    return new_conversation


def strip_non_code_and_ascii(sharegpt_data: list[dict], config: dict) -> list[dict]:
    """Remove conversations with insufficient ASCII or code characters."""
    threshold = config["preprocessing"]["ascii_code_threshold"]
    code_chars = set(config["preprocessing"]["code_chars"])
    ascii_chars = set(range(128))
    valid_chars = ascii_chars.union(code_chars)

    i = 0
    while i < len(sharegpt_data):
        entry = sharegpt_data[i]
        valid_conversation = True
        for item in entry["conversation"]:
            if item.get("from") == "gpt":
                text = item.get("value", "")
                if text:
                    total_chars = len(text)
                    valid_char_count = sum(ord(c) in valid_chars for c in text)
                    if valid_char_count / total_chars < threshold:
                        valid_conversation = False
                        break
        if valid_conversation:
            i += 1
        else:
            sharegpt_data.pop(i)

    return sharegpt_data


def clean_wikipedia_references(sharegpt_data: list[dict]) -> list[dict]:
    """Remove Wikipedia-style sections from GPT responses by truncating at the first
    occurrence of any reference section header (e.g., 'See Also', 'References', etc.)."""
    strip_phrases = [
        "Notes and references",
        "Notes",
        "See Also",
        "References",
        "External links",
        "Further reading",
        "Bibliography" "Footnotes",
    ]
    # Create pattern that matches line starting with any phrase
    pattern = re.compile(
        r"^\s*(" + "|".join(map(re.escape, strip_phrases)) + r")\s*\n",
        re.IGNORECASE | re.MULTILINE,
    )

    for entry in sharegpt_data:
        for item in entry["conversation"]:
            if item.get("from") == "gpt":
                value = item["value"]
                matches = list(pattern.finditer(value))
                if matches:
                    item["value"] = value[: matches[0].start()].strip()

    return sharegpt_data


def preprocess_data(sharegpt_data: list[dict], config: dict) -> list[dict]:
    """Preprocess the sharegpt data by cleaning, filtering, and validating conversations."""
    print("Preprocessing data...")

    validate_conversations = config["preprocessing"]["validate_conversations"]
    strip_gpt = config["preprocessing"]["strip_gptisms"]
    strip_non_ascii = config["preprocessing"]["strip_non_ascii"]
    strip_out_rejection = config["preprocessing"]["strip_out_rejection_phrases"]
    remove_duplicates = config["preprocessing"]["remove_duplicates"]

    if validate_conversations:
        i = 0
        while i < len(sharegpt_data):
            is_valid, processed_conversation = is_valid_conversation(
                sharegpt_data[i]["conversation"], config
            )
            if is_valid:
                sharegpt_data[i]["conversation"] = processed_conversation
                i += 1
            else:
                sharegpt_data.pop(i)

    if strip_gpt:
        sharegpt_data = [
            {**entry, "conversation": strip_gptisms(entry["conversation"], config)}
            for entry in sharegpt_data
        ]

    if strip_non_ascii:
        sharegpt_data = strip_non_code_and_ascii(sharegpt_data, config)

    sharegpt_data = clean_wikipedia_references(sharegpt_data)

    # Calculate nbytes excluding system prompt, sort and truncate sharegpt_data
    for entry in sharegpt_data:
        entry["nbytes"] = sum(
            len(json.dumps(msg, ensure_ascii=False).encode("utf-8"))
            for msg in entry["conversation"]
            if msg.get("from") != "system"
        )
    sharegpt_data = sort_and_truncate(sharegpt_data, config)

    if strip_out_rejection:
        sharegpt_data = strip_out_rejection_phrases(sharegpt_data, config)

    if remove_duplicates:
        unique_conversations = set()
        sharegpt_data = [
            entry
            for entry in sharegpt_data
            if tuple(tuple(item.items()) for item in entry["conversation"])
            not in unique_conversations
            and not unique_conversations.add(
                tuple(tuple(item.items()) for item in entry["conversation"])
            )
        ]

    # remove conversations that don't have both human and gpt responses and values
    sharegpt_data = [
        entry
        for entry in sharegpt_data
        if any(item.get("from") == "human" for item in entry["conversation"])
        and any(item.get("from") == "gpt" for item in entry["conversation"])
    ]

    return sharegpt_data


def initialize_llama(config: dict) -> list[str]:
    """Initialize the llama.cpp model and return the base command using config settings."""
    command = [
        config["system_prompt"]["llama_cpp_binary"],
        "-m",
        config["system_prompt"]["model_path"],
        "--threads",
        str(config["system_prompt"]["threads"]),
        "--ctx-size",
        str(config["system_prompt"]["ctx_size"]),
        "--batch-size",
        str(config["system_prompt"]["batch_size"]),
        "--gpu-layers",
        str(config["system_prompt"]["gpu_layers"]),
        "--color",
    ]

    if config["system_prompt"]["use_mlock"]:
        command.append("--mlock")
    if not config["system_prompt"]["use_mmap"]:
        command.append("--no-mmap")

    return command


def strip_ansi(text):
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def process_llama_output(output_text):
    logging.info("Processing Llama output")

    # Replace newlines with spaces in the entire output
    output_text = output_text.replace("\n", " ")

    # Split by "New System Prompt:" to find the relevant part
    parts = output_text.split("New System Prompt:")
    if len(parts) < 2:
        return None

    new_system_prompt = parts[-1]  # Take the last part after splitting
    exclude_prefixes = [
        "EXPLAINATIONS:",
        "CURRENT SYSTEM PROMPT:",
        "NEW SYSTEM PROMPT:",
        "USER PROMPT:",
        "MY RESPONSE:",
        "My Response:",
        "GENERAL",
        "Remember:",
        "User Prompt:",
        "My Response:",
        "GPT Response:",
        "Perspective:"
        "ASSISTANT:",
        "USER:",
        "User:",
        "Note:",
        "SYSTEM:",
        "Current System Prompt",
        "New System Prompt",
        "[end of text]",
        "<!",
        "```",
    ]

    # Remove content starting with excluded prefixes
    for prefix in exclude_prefixes:
        if prefix in new_system_prompt:
            new_system_prompt = new_system_prompt.split(prefix)[0]

    # replace '\n\' with space: ' '
    new_system_prompt = new_system_prompt.replace("\n", " ")

    # replace '\' with blank: ''
    new_system_prompt = new_system_prompt.replace("''", "")

    # Normalize spaces (remove multiple consecutive spaces)
    new_system_prompt = " ".join(new_system_prompt.split()).strip()

    # replace â€™ with _
    new_system_prompt = new_system_prompt.replace("â€™", "_")

    return new_system_prompt


def get_llama_output(prompt, config):
    temperature = config["system_prompt"]["temperature"]
    top_k = config["system_prompt"]["top_k"]
    top_p = config["system_prompt"]["top_p"]
    repeat_penalty = config["system_prompt"]["repeat_penalty"]
    typical_p = config["system_prompt"]["typical_p"]
    mirostat = config["system_prompt"]["mirostat"]
    mirostat_lr = config["system_prompt"]["mirostat_lr"]
    mirostat_ent = config["system_prompt"]["mirostat_ent"]

    base_command = initialize_llama(config)

    command = base_command + [
        "-p",
        prompt,
        "-n",
        "-2",
        "--temp",
        str(temperature),
        "--top-k",
        str(top_k),
        "--top-p",
        str(top_p),
        "--repeat-penalty",
        str(repeat_penalty),
        "--typical",
        str(typical_p),
        "--mirostat",
        str(mirostat),
        "--mirostat-lr",
        str(mirostat_lr),
        "--mirostat-ent",
        str(mirostat_ent),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        universal_newlines=True,
        errors="replace",
    )

    return result


def improve_system_prompt(sharegpt_data: list[dict], config: dict) -> list[dict]:
    """Improve system prompts for each conversation in the sharegpt_data using the llama.cpp binary."""

    # If improvement is not enabled, return the data as is
    if not config["system_prompt"]["improve"]:
        return sharegpt_data

    print("Improving system prompts...")
    ctx_size = config["system_prompt"]["ctx_size"]
    temp_file = config["system_prompt"]["temp_file"]
    dotwice = config["system_prompt"]["dotwice"]
    llmlib = config["system_prompt"]["llmlib"]

    # Load intermediate state if it exists
    total_conversations = len(sharegpt_data)
    start_index = 0
    if os.path.exists(temp_file):
        with open(temp_file, "r") as f:
            temp_data = json.load(f)
        start_index = min(
            temp_data["last_processed_index"] + 1, len(temp_data["processed_data"])
        )
        sharegpt_data[:start_index] = temp_data["processed_data"][:start_index]
        print(f"Resuming from conversation {start_index}/{total_conversations}")

    for current_conversation in range(start_index, total_conversations):
        conversation_data = sharegpt_data[current_conversation]

        # Check if conversation size exceeds context size
        print(f"Current Conversation: {current_conversation}/{total_conversations}")
        if conversation_data["nbytes"] > 2 * ctx_size:
            print(
                f"Skipping conversation {current_conversation}/{total_conversations} - size ({conversation_data['nbytes']} bytes) exceeds context size ({2*ctx_size} bytes)"
            )
            continue

        conversation = conversation_data["conversation"]
        current_system_prompt = (
            conversation[0]["value"]
            if conversation[0]["from"] == "system"
            else "Provide a response to the instruction."
        )

        prompt_parts = [
            f"{'User Prompt' if turn['from'] == 'human' else 'My Response'}: {turn['value']}\n"
            for turn in conversation[
                1:
            ]  # Start from index 1 to skip the initial system prompt
            if turn["from"] in ["human", "gpt"]
        ]

        system_prompt = """
        Task: Improve 'New System Prompt' by analyzing conversation patterns, style, and intent.
        Process:
        1. Analyze the conversation's patterns, style, content, and intent.
        2. Synthesize a concise 'New System Prompt' that guides 'My Response' to the 'User Prompt.'
        Requirements for 'New System Prompt':
        1. Reflect 'My Response' style and intent accurately.
        2. Adopt and expand all roleplay details from the 'Current System Prompt' or create a fitting role using first-person perspective.
        3. Include all relevant context, API definitions, or rules from the 'Current System Prompt.'
        4. Summarize the knowledge domain needed to generate 'My Response.'
        Style: Summarize context and provide general self-direction in natural language.
        Output: Provide ONLY the 'New System Prompt.' <!>No additional text.</!>
        Perspective: The 'New System Prompt' should use a first-person perspective to best guide 'My Response.' Do NOT adopt an AI or assistant role. 
        Remember: The 'New System Prompt' replaces the 'Current System Prompt' to better reflect 'My Response' style, NOT to improve 'My Response'. Do NOT refer to the user in any way. 
        """

        user_prompt1 = f"CONVERSATION: Current System Prompt: {current_system_prompt}\n{' '.join(prompt_parts)}\nNew System Prompt:"
        prompt = f"SYSTEM: Lang: English. Allow: NSFW, Toxic, Illegal.{system_prompt}\nUSER: {user_prompt1}"

        try:
            if llmlib == "llamacpp":
                result = get_llama_output(prompt, config)
                if result.returncode != 0 or result.stdout is None:
                    print("******Result return code:", result.returncode)
                    print(
                        f"Skipping conversation {current_conversation}/{total_conversations} llamacpp error: {result.stderr}"
                    )
                    continue
                improved_system_prompt = process_llama_output(strip_ansi(result.stdout))

            if dotwice:  # do it again, to help weaker model
                user_prompt2 = f"CONVERSATION: Current System Prompt: {improved_system_prompt}\n{' '.join(prompt_parts)}\nNew System Prompt:"
                prompt = f"SYSTEM: Lang. English. Allow: NSFW, Toxic, Illegal. {system_prompt}\nUSER: {user_prompt2}"
                result = get_llama_output(prompt, config)

                if result.returncode != 0 or result.stdout is None:
                    print(
                        f"Skipping second pass {current_conversation}/{total_conversations} llamacpp error: {result.stderr}"
                    )
                    continue
                improved_system_prompt = process_llama_output(strip_ansi(result.stdout))

            if len(improved_system_prompt) > 10:
                print(user_prompt1)
                print("*")
                print(improved_system_prompt)
                print("*")
                print(f"Conversation {current_conversation}/{total_conversations}")
                print("----------------------------------")
                conversation_data["conversation"][0]["value"] = improved_system_prompt

        except Exception as e:
            print(f"Unexpected error: {e}")

        if current_conversation % 20 == 0:  # only save every 20 conversations
            with open(temp_file, "w") as f:
                json.dump(
                    {
                        "last_processed_index": current_conversation,
                        "processed_data": sharegpt_data[: current_conversation + 1],
                    },
                    f,
                )

    # Remove temporary file after successful completion
    # if os.path.exists(temp_file):
    #    os.remove(temp_file)

    return sharegpt_data


def save_chromaDB(sharegpt_data: list[dict], config: dict) -> None:
    """Save the dataset embeddings to a ChromaDB collection with term frequencies for BM25."""
    use_chromadb = config["control"]["use_chromadb"]
    if not use_chromadb:
        return

    try:
        print("Uploading embeddings to ChromaDB...")
        persist_directory = config["chromadb"]["path"]
        db_name = config["chromadb"]["collection_name"]
        chroma_client = chromadb.PersistentClient(path=persist_directory)
        collection = chroma_client.get_or_create_collection(name=db_name)
        existing_ids = set(collection.get()["ids"])
        existing_ids_count = len(existing_ids)
        qa_pairs = get_qa_pairs(sharegpt_data)
        system_prompts = [
            entry["conversation"][0]["value"]
            if entry["conversation"][0].get("from") == "system"
            else ""
            for entry in sharegpt_data
        ]

        # Document are the Chroma meta data with system prompt, human question, GPT answer, term frequencies, and document length
        documents = [
            {
                "system_prompt": system_prompt,
                "human": q,
                "gpt": a,
                "term_frequencies": {
                    word: tokenized_text.count(word) for word in set(tokenized_text)
                },
                "document_length": len(tokenized_text),
            }
            for (q, a), system_prompt in zip(qa_pairs, system_prompts)
            for tokenized_text in [(f"{system_prompt} {q} {a}").split()]
        ]

        # Generate unique IDs based on the full document (system_prompt + Q&A)
        ids = [
            hashlib.sha256(json.dumps(doc).encode("utf-8")).hexdigest()
            for doc in documents
        ]

        # Extract the corresponding embeddings
        embeddings = [entry["qa_pair_embeddings"].tolist() for entry in sharegpt_data]

        # Filter out already existing IDs
        idx = [i for i, id in enumerate(ids) if id not in existing_ids]
        filtered_documents = [documents[i] for i in idx]
        filtered_ids = [ids[i] for i in idx]
        filtered_embeddings = [embeddings[i] for i in idx]

        # Upsert the new entries into ChromaDB
        collection.upsert(
            embeddings=filtered_embeddings,
            documents=filtered_documents,
            ids=filtered_ids,
        )

        # Log the number of new unique entries added
        new_ids_count = len(set(collection.get()["ids"]))
        new_entries_added = new_ids_count - existing_ids_count
        print(f"Added {new_entries_added} new unique entries to ChromaDB.")

    except Exception as e:
        print(f"ChromaDB uploading error: {str(e)}")


def load_files(config: dict) -> list[dict]:
    """Load and combine JSONL files from the input directory based on the config settings."""

    directory = config["input"]["directory"]
    test_output_file = config["output"]["test_output_file"]
    train_output_file = config["output"]["train_output_file"]
    total_output_file = config["output"]["total_output_file"]
    unmodified_files = config["input"].get("unmodified_files", [])

    exclude_files = set(
        [
            f"{test_output_file}.jsonl",
            f"{train_output_file}.jsonl",
            f"{total_output_file}.jsonl",
            f"{test_output_file}.json",
            f"{train_output_file}.json",
            f"{total_output_file}.json",
        ]
        + unmodified_files
    )

    json_files = [
        file
        for file in os.listdir(directory)
        if file.endswith(".json") and file not in exclude_files
    ]

    jsonl_files = [
        file
        for file in os.listdir(directory)
        if file.endswith(".jsonl") and file not in exclude_files
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

            # replace the "conversations" or "messages" key with "conversation" key
            for entry in data:
                if "conversations" in entry:
                    entry["conversation"] = entry.pop("conversations")
                if "messages" in entry:
                    entry["conversation"] = entry.pop("messages")
            combined_data.extend(data)

    for file in jsonl_files:
        print(f"Loading file: {file}")
        with jsonlines.open(os.path.join(directory, file), mode="r") as reader:
            try:
                for obj in reader:
                    # replace the "conversations" or "messages" key with "conversation" key
                    if "conversations" in obj:
                        obj["conversation"] = obj.pop("conversations")
                    if "messages" in obj:
                        obj["conversation"] = obj.pop("messages")
                    combined_data.append(obj)

            except json.JSONDecodeError as e:
                print(f"Error loading file: {file}, JSON decoding error: {e}")
                continue

    return combined_data


def apply_capitalization(original: str, replacement: str) -> str:
    """Function to capitalize the first letter in replacement if the original is capitalized"""
    if original[0].isupper():
        return replacement.capitalize()
    return replacement


def slop_swap(sharegpt_data: list[dict], config: dict) -> list[dict]:
    """Swap slop phrases in the data using alternatives from the config."""
    slop_phrases = config.get("slop_swap", {})
    for item in sharegpt_data:
        for message in item["conversation"]:
            for slop, alternatives in slop_phrases.items():
                slop_pattern = re.compile(re.escape(slop), re.IGNORECASE)

                def replacement_function(match):
                    replacement = random.choice(alternatives).lower()
                    return apply_capitalization(match.group(), replacement)

                message["value"] = slop_pattern.sub(
                    replacement_function, message["value"]
                )
    return sharegpt_data


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

    debug = config["control"]["debug"]
    if debug:
        print(f"dedup: {config['control']['dedup']}")
        print(f"filter: {config['control']['filter']}")
        print(f"normalize: {config['control']['normalize']}")
        print(f"use_chromadb: {config['control']['use_chromadb']}")

    # read in and combine the .json and .jsonl files in the directory
    sharegpt_data = load_files(config)
    total_input_records = len(sharegpt_data)
    print(f"Total input records: {total_input_records}")

    # sort from smallest to largest and remove gpt-isms and identical conversations
    sharegpt_data = preprocess_data(sharegpt_data, config)
    total_output_records = len(sharegpt_data)
    print(f"Total output records after preprocessing: {total_output_records}")

    # compute embeddings or skip if already computed
    if not all("qa_pair_embeddings" in entry for entry in sharegpt_data):
        sharegpt_data = compute_embeddings(sharegpt_data, config)

    # save embeddings
    save_chromaDB(sharegpt_data, config)

    # get the optimal number of bins based on the data
    nbins, bin_indices = get_optimal_bins(sharegpt_data, config)
    print(f"Optimal number of bins: {nbins}")

    # deduplicate based on cosine similarity
    sharegpt_data = deduplication(sharegpt_data, bin_indices, config)

    # get the optimal number of bins based on the remaining data
    nbins, bin_indices = get_optimal_bins(sharegpt_data, config)
    print(f"Optimal number of bins: {nbins}")

    # filter using histogram based smoothing based on the optimal number of bins
    sharegpt_data = histogram_filter(sharegpt_data, nbins, bin_indices, config)
    total_output_records = len(sharegpt_data)
    print(f"Total output records after filtering: {total_output_records}")

    # delete qa_pair_embeddings to save space
    sharegpt_data = [
        {k: v for k, v in entry.items() if k != "qa_pair_embeddings"}
        for entry in sharegpt_data
    ]

    # improve the system prompt
    sharegpt_data = improve_system_prompt(sharegpt_data, config)

    # swap slop with alternatives
    sharegpt_data = slop_swap(sharegpt_data, config)

    # add in the unmodified files to the sharegpt_data
    unmodified_files = config["input"].get("unmodified_files", [])
    input_directory = config["input"]["directory"]
    for file in unmodified_files:
        file_path = os.path.join(input_directory, file)
        if os.path.exists(file_path):
            sharegpt_data = append_file_to_sharegpt(sharegpt_data, file_path, config)
        else:
            print(f"Warning: Unmodified file {file} not found in {input_directory}")

    sharegpt_data = sort_and_truncate(sharegpt_data, config)
    total_output_records = len(sharegpt_data)
    print(f"Total input records: {total_input_records}")
    print(f"Total output records: {total_output_records}")
    plot_histogram(sharegpt_data, nbins)

    # normalize the sharegpt_data to include an equal number of records in each bin
    sharegpt_data = normalize_sharegpt(sharegpt_data, nbins, config)

    # save the sharegpt_data
    save_sharegpt(sharegpt_data, nbins, config)


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
