"""
Data preparation script for rubric-based RL training.

Downloads datasets from HuggingFace and converts to verl format.
Supports: RaR-Science, RaR-Medicine, HealthBench.

Environment variables:
    RUBRIC_DATASET: "science", "medicine", or "healthbench" (default: "science")
    OUTPUT_DIR: Where to save parquet files
    PRIVILEGED_INFO_MODE: "rubric", "reference", or "rubric_and_reference" (default: "rubric")
"""

import os
import random
import argparse
from typing import Dict, Any, List

import pandas as pd
from datasets import load_dataset


def format_rubric_privileged(rubric):
    """Format rubric into privileged info text.

    Args:
        rubric: List of dicts with keys: title, description, weight

    Returns:
        Formatted rubric text string
    """
    if not rubric:
        return ""

    lines = []

    for item in rubric:
        title = item.get("title", "")
        description = item.get("description", "")
        weight = item.get("weight", 1.0)
        if weight is None:
            weight = 1.0

        if weight > 0:
            lines.append(f"  - {title}: {description} (weight: +{weight})")
        else:
            lines.append(f"  - {title}: {description} (weight: {weight})")

    rubric_text = "\n".join(lines)

    instruction = "You should follow the given rubrics to answer the question."
    instruction += " Positive weights indicate criteria you should satisfy."
    instruction += " Negative weights indicate criteria you should avoid."

    return f"{instruction}\n\n{rubric_text}"


def format_reference_privileged(reference_answer):
    """Format reference answer into privileged info text.

    Args:
        reference_answer: The reference answer string

    Returns:
        Formatted reference text string
    """
    if not reference_answer:
        return ""

    return f"You should refer to the following reference answer:\n\n{reference_answer}"


def format_rubric_and_reference_privileged(rubric, reference_answer):
    """Format both rubric and reference answer into privileged info text.

    Args:
        rubric: List of dicts with keys: title, description, weight
        reference_answer: The reference answer string

    Returns:
        Formatted combined text string
    """
    parts = []

    rubric_text = format_rubric_privileged(rubric)
    if rubric_text:
        parts.append(rubric_text)

    reference_text = format_reference_privileged(reference_answer)
    if reference_text:
        parts.append(reference_text)

    return "\n\n".join(parts)


def convert_healthbench_rubric(hb_rubric: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert HealthBench rubric format to internal rubric format.

    HealthBench format: {"criterion": str, "points": int, "tags": ["axis:X", ...]}
    Internal format: {"title": str, "description": str, "weight": float}

    Args:
        hb_rubric: List of HealthBench rubric items

    Returns:
        List of internal rubric items
    """
    result = []
    for item in hb_rubric:
        # Extract axis from tags for title
        title = "criterion"
        for tag in item.get("tags", []):
            if tag.startswith("axis:"):
                title = tag[len("axis:"):]
                break

        result.append({
            "title": title,
            "description": item["criterion"],
            "weight": float(item.get("points", 1)),
        })
    return result


def format_multiturn_question(prompt: List[Dict[str, str]]) -> str:
    """Format a multi-turn conversation into a readable text block for judge context.

    For single-turn prompts, returns just the user content.
    For multi-turn, formats as "Role: content" lines.

    Args:
        prompt: List of message dicts with "role" and "content" keys

    Returns:
        Formatted conversation string
    """
    if len(prompt) == 1:
        return prompt[0]["content"]

    lines = []
    for msg in prompt:
        role = msg["role"].capitalize()
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def deterministic_split(
    data: List[Dict[str, Any]],
    train: int,
    val: int,
    test: int,
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """Deterministically shuffle and split data into train/val/test.

    Args:
        data: List of examples to split
        train: Number of training examples
        val: Number of validation examples
        test: Number of test examples
        seed: Random seed for reproducibility

    Returns:
        Dict with "train", "val", "test" keys mapping to lists of examples

    Raises:
        ValueError: If train + val + test != len(data)
    """
    if train + val + test != len(data):
        raise ValueError(
            f"Split sizes ({train}+{val}+{test}={train+val+test}) "
            f"don't match data size ({len(data)})"
        )

    shuffled = list(data)
    random.Random(seed).shuffle(shuffled)

    return {
        "train": shuffled[:train],
        "val": shuffled[train:train + val],
        "test": shuffled[train + val:],
    }


def load_healthbench() -> List[Dict[str, Any]]:
    """Load HealthBench dataset and convert to internal format.

    Downloads openai/healthbench from HuggingFace, extracts the oss_eval split,
    and converts each row to our internal format with:
    - prompt: multi-turn conversation (List[Dict])
    - question: formatted text for judge context
    - rubric: List[Dict] with title/description/weight
    - reference_answer: ideal completion text

    Returns:
        List of examples in internal format (before VERL conversion)
    """
    # Load as raw JSON to avoid HF schema mismatch with the dataset's metadata
    raw_data = load_dataset(
        "json",
        data_files="hf://datasets/openai/healthbench/2025-05-07-06-14-12_oss_eval.jsonl",
        split="train",
    )

    examples = []
    for row in raw_data:
        prompt = row["prompt"]
        question = format_multiturn_question(prompt)
        rubric = convert_healthbench_rubric(row["rubrics"])
        ideal_data = row.get("ideal_completions_data")
        reference_answer = ""
        if ideal_data and isinstance(ideal_data, dict):
            reference_answer = ideal_data.get("ideal_completion", "") or ""

        examples.append({
            "prompt": prompt,
            "question": question,
            "rubric": rubric,
            "reference_answer": reference_answer,
        })

    return examples


def convert_to_verl_format(
    dataset_name: str,
    split: str,
    data: List[Dict[str, Any]],
    privileged_info_mode: str = "rubric",
) -> List[Dict[str, Any]]:
    """Convert RaR dataset to verl RLHFDataset format.

    Required columns for verl:
    - prompt: list of message dicts [{"role": "user", "content": <question>}]
    - data_source: "rubric_rl"
    - ability: "reasoning"
    - reward_model: {"style": "rule", "ground_truth": <reference_answer>}
    - extra_info: dict with question, rubric, split, reference_answer, priv_info

    Args:
        dataset_name: "science" or "medicine"
        split: "train" or "val"
        data: List of dataset examples
        privileged_info_mode: Which privileged info to use
            - "rubric": Only use formatted rubric
            - "reference": Only use reference answer
            - "rubric_and_reference": Use both

    Returns:
        List of examples in verl format
    """
    verl_examples = []

    for example in data:
        # Build prompt in chat format
        if "prompt" in example and isinstance(example["prompt"], list):
            # Multi-turn: use prompt directly (e.g., HealthBench)
            prompt = example["prompt"]
            question_for_judge = format_multiturn_question(prompt)
        else:
            # Single-turn: wrap question in chat format (e.g., RaR)
            prompt = [{"role": "user", "content": example["question"]}]
            question_for_judge = example["question"]

        # Get reference answer
        reference_answer = example.get("reference_answer", "")

        # Get rubric (preserve full rubric for reward function)
        rubric = example.get("rubric", [])
        if isinstance(rubric, str):
            # Parse rubric if it's a string
            rubric = []

        # Format privileged info based on mode
        if privileged_info_mode == "rubric":
            priv_info = format_rubric_privileged(rubric)
        elif privileged_info_mode == "reference":
            priv_info = format_reference_privileged(reference_answer)
        elif privileged_info_mode == "rubric_and_reference":
            priv_info = format_rubric_and_reference_privileged(rubric, reference_answer)
        else:
            priv_info = ""

        # Build extra_info
        extra_info = {
            "question": question_for_judge,
            "rubric": rubric,
            "split": split,
            "reference_answer": reference_answer,
            "priv_info": priv_info,  # Trainer directly uses this
            "messages": prompt,  # Original chat messages for multi-turn distillation
        }

        # Build verl example
        verl_example = {
            "prompt": prompt,
            "data_source": "rubric_rl",
            "ability": "reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": reference_answer,
            },
            "extra_info": extra_info,
        }

        verl_examples.append(verl_example)

    return verl_examples


def prepare_dataset(
    dataset_name: str = "science",
    output_dir: str = "./data/rubric_rl",
    privileged_info_mode: str = "rubric",
) -> None:
    """
    Download and convert RaR dataset to verl format.

    Args:
        dataset_name: "science" or "medicine"
        output_dir: Directory to save parquet files
        privileged_info_mode: "rubric", "reference", or "rubric_and_reference"
    """
    # Validate privileged_info_mode
    if privileged_info_mode not in ["rubric", "reference", "rubric_and_reference"]:
        raise ValueError(
            f"Unknown privileged_info_mode: {privileged_info_mode}. "
            "Use 'rubric', 'reference', or 'rubric_and_reference'."
        )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    if dataset_name == "healthbench":
        # HealthBench has a single oss_eval split — load, convert, and split
        print("Loading HealthBench dataset (oss_eval split)...")
        print(f"Using privileged_info_mode: {privileged_info_mode}")
        examples = load_healthbench()
        print(f"Loaded {len(examples)} examples from HealthBench oss_eval")

        splits = deterministic_split(examples, train=4000, val=500, test=500, seed=42)

        for split_name, split_data in splits.items():
            print(f"Processing {split_name} split ({len(split_data)} examples)")
            verl_examples = convert_to_verl_format(
                dataset_name=dataset_name,
                split=split_name,
                data=split_data,
                privileged_info_mode=privileged_info_mode,
            )
            df = pd.DataFrame(verl_examples)
            output_path = os.path.join(output_dir, f"{split_name}.parquet")
            df.to_parquet(output_path, index=False)
            print(f"Saved {len(df)} examples to {output_path}")

        print(f"Dataset preparation complete. Files saved to {output_dir}")
        return

    # Map to HuggingFace dataset name for RaR datasets
    hf_datasets = {
        "science": "anisha2102/RaR-Science",
        "medicine": "anisha2102/RaR-Medicine",
    }

    if dataset_name not in hf_datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'science', 'medicine', or 'healthbench'.")

    dataset_path = hf_datasets[dataset_name]
    print(f"Loading dataset: {dataset_path}")
    print(f"Using privileged_info_mode: {privileged_info_mode}")

    # Load dataset from HuggingFace
    dataset = load_dataset(dataset_path)

    # Process each split
    for split_name in ["train", "val", "test"]:
        if split_name not in dataset:
            continue

        # Map split name to our convention
        if split_name == "val":
            our_split = "val"
        else:
            our_split = split_name

        print(f"Processing {split_name} split ({len(dataset[split_name])} examples)")

        # Convert to verl format
        verl_examples = convert_to_verl_format(
            dataset_name=dataset_name,
            split=our_split,
            data=dataset[split_name],
            privileged_info_mode=privileged_info_mode,
        )

        # Save as parquet
        df = pd.DataFrame(verl_examples)
        output_path = os.path.join(output_dir, f"{our_split}.parquet")
        df.to_parquet(output_path, index=False)
        print(f"Saved {len(df)} examples to {output_path}")

    print(f"Dataset preparation complete. Files saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare RaR dataset for rubric RL training"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.environ.get("RUBRIC_DATASET", "science"),
        choices=["science", "medicine", "healthbench"],
        help="Dataset to download (default: science)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("OUTPUT_DIR", "./data/rubric_rl"),
        help="Output directory for parquet files",
    )
    parser.add_argument(
        "--privileged_info_mode",
        type=str,
        default=os.environ.get("PRIVILEGED_INFO_MODE", "rubric"),
        choices=["rubric", "reference", "rubric_and_reference"],
        help="Which privileged info to use (default: rubric)",
    )

    args = parser.parse_args()

    prepare_dataset(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        privileged_info_mode=args.privileged_info_mode,
    )


if __name__ == "__main__":
    main()
