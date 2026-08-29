import concurrent.futures
import os
import json
import traceback
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env
# Try to find .env starting from the script's directory and going up
POINT_RUBRIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(POINT_RUBRIC_DIR, "..", ".env"))

# Configuration
DATA_DIR = os.path.join(POINT_RUBRIC_DIR, "data")
INPUT_FILE = os.path.join(DATA_DIR, "retrieved_questions.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "retrieved_questions_refined_rubrics.json")
PROMPT_FILE = os.path.join(POINT_RUBRIC_DIR, "rubric_prompts", "refine_rubric.txt")
MODEL = "gpt-4o"
MAX_RETRIES = 3
MAX_WORKERS = 8 # Number of parallel API calls

def load_prompt_template():
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read()

def validate_refined_rubric(refined_json):
    """
    Validates that the refined rubric has exactly 6 rules: 2 hard and 4 soft.
    """
    if not isinstance(refined_json, list):
        return False, "Output is not a list."

    if len(refined_json) != 6:
        return False, f"Expected 6 rules, but got {len(refined_json)}."

    hard_count = sum(1 for rule in refined_json if rule.get("type") == "hard")
    soft_count = sum(1 for rule in refined_json if rule.get("type") == "soft")

    if hard_count != 2 or soft_count != 4:
        return False, f"Expected 2 hard and 4 soft rules, but got {hard_count} hard and {soft_count} soft."

    return True, ""

def refine_rubrics():
    # Initialize the client
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI-KEY")
    org_id = os.getenv("OPENAI_ORG_ID")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment or .env file.")
        return
    client = OpenAI(
        api_key=api_key,
        organization=org_id,
        timeout=120.0  # Increased timeout for rubric refinement
    )

    # Load questions
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    prompt_template = load_prompt_template()
    refined_questions = [None] * len(questions) # Placeholder to preserve order

    print(f"Starting refinement for {len(questions)} questions using {MODEL} with {MAX_WORKERS} workers...")

    def process_item(index, item):
        question_id = item["question_id"]
        original_rubric = item["rubric"]
        question_text = item["question"]

        print(f"[{index+1}/{len(questions)}] Refining rubric for question ID: {question_id}...")

        success = False
        attempts = 0
        refined_data = None

        while not success and attempts < MAX_RETRIES:
            attempts += 1
            try:
                formatted_prompt = prompt_template.replace("{question_id}", str(question_id)).replace("{rubric}", original_rubric)

                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "You are a precise and analytical editor specializing in evaluation rubrics."},
                        {"role": "user", "content": formatted_prompt}
                    ],
                    temperature=0.3
                )

                content = response.choices[0].message.content.strip()
                if content.startswith("```json"):
                    content = content[len("```json"):].strip()
                if content.endswith("```"):
                    content = content[:-3].strip()

                try:
                    refined_json = json.loads(content)
                    is_valid, error_msg = validate_refined_rubric(refined_json)
                    if is_valid:
                        refined_data = refined_json
                        success = True
                    else:
                        print(f"  [{question_id}] Attempt {attempts}: Validation failed - {error_msg}")
                except json.JSONDecodeError:
                    print(f"  [{question_id}] Attempt {attempts}: Failed to parse JSON.")

            except Exception as e:
                print(f"  [{question_id}] Attempt {attempts}: Error during API call: {e}")

        if success:
            return {
                "question_id": question_id,
                "question": question_text,
                "original_rubric": original_rubric,
                "refined_rubric": refined_data
            }
        else:
            print(f"  FAILED after {MAX_RETRIES} attempts for question ID: {question_id}")
            return None

    # Run in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_item, i, item): i for i, item in enumerate(questions)}
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            res = future.result()
            if res:
                refined_questions[idx] = res

    # Filter out FAILED ones
    final_results = [q for q in refined_questions if q is not None]

    # Save results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)

    print(f"\nRefinement complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    refine_rubrics()
