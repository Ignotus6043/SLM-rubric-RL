import sys
import os
import json
import random
from datasets import load_dataset

# Configuration
# Save the Hugging Face cache within the project directory for portability.
POINT_RUBRIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_DIR = os.path.join(POINT_RUBRIC_DIR, "hf_cache")
DATA_DIR = os.path.join(POINT_RUBRIC_DIR, "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "retrieved_questions.json")

def retrieve_questions(num_samples=10, seed=42):
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Using cache directory: {CACHE_DIR}")
    print(f"Using random seed: {seed}")
    print("Loading OpenRubrics dataset...")

    # Try to load the dataset (mirroring the logic in OpenRubrics/eval.py)
    try:
        dataset = load_dataset("OpenRubrics/OpenRubric-v2", split="train", cache_dir=CACHE_DIR)
    except Exception as e1:
        print(f"Could not load OpenRubric-v2 ({e1}), trying fallback dataset...")
        try:
            dataset = load_dataset("OpenRubrics/OpenRubrics", split="train", cache_dir=CACHE_DIR)
        except Exception as e2:
            print(f"Error: Could not load either dataset. {e2}")
            return

    # Ensure we don't try to sample more than available
    total_available = len(dataset)
    print(f"Filtering dataset for questions with exactly 2 hard rules...")

    # For efficiency and to maintain random sampling while filtering
    random.seed(seed)
    shuffled_indices = list(range(total_available))
    random.shuffle(shuffled_indices)

    retrieved_data = []

    for idx in shuffled_indices:
        if len(retrieved_data) >= num_samples:
            break

        item = dataset[idx]
        rubric = item.get("rubric", "")

        # Count occurrences of "[Hard Rule]" in the rubric
        hard_rule_count = rubric.count("[Hard Rule]")

        if hard_rule_count == 2:
            # Extract fields
            question_id = str(item.get("id", item.get("uuid", idx)))
            question = item.get("instruction", item.get("prompt", ""))

            # Create JSON object
            retrieved_data.append({
                "question_id": question_id,
                "question": question,
                "rubric": rubric
            })

    print(f"Filtering complete. Found {len(retrieved_data)} questions meeting the criteria.")

    # Save to file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(retrieved_data, f, indent=4, ensure_ascii=False)

    print(f"Successfully retrieved and saved {len(retrieved_data)} questions to {OUTPUT_FILE}")

if __name__ == "__main__":
    num_samples = 10
    seed = 42

    if len(sys.argv) > 1:
        try:
            num_samples = int(sys.argv[1])
        except ValueError:
            print(f"Warning: Could not parse '{sys.argv[1]}' as an integer. Using default value of 10.")

    if len(sys.argv) > 2:
        try:
            seed = int(sys.argv[2])
        except ValueError:
            print(f"Warning: Could not parse '{sys.argv[2]}' as an integer for seed. Using default value of 42.")

    retrieve_questions(num_samples, seed)
