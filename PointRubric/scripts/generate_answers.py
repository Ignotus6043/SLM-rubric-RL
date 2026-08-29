import concurrent.futures
import os
import json
import time
import traceback
from openai import OpenAI
from dotenv import load_dotenv

# ... Configuration ...
POINT_RUBRIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(POINT_RUBRIC_DIR, "data")
INPUT_FILE = os.path.join(DATA_DIR, "retrieved_questions_refined_rubrics.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "generated_answers.json")
MAX_WORKERS = 5 # Number of parallel questions being processed

# Prompt files
PROMPT_GOLD = os.path.join(POINT_RUBRIC_DIR, "answer_prompts", "gold.txt")
PROMPT_BAD = os.path.join(POINT_RUBRIC_DIR, "answer_prompts", "bad.txt")
PROMPT_PARTIAL = os.path.join(POINT_RUBRIC_DIR, "answer_prompts", "partial.txt")
PROMPT_REASON = os.path.join(POINT_RUBRIC_DIR, "answer_prompts", "reason.txt")
PROMPT_PLAIN = os.path.join(POINT_RUBRIC_DIR, "answer_prompts", "plain.txt")

def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def format_rubric_string(refined_rubric_list):
    """
    Converts the list of refined rubric objects into a numbered string for the prompt.
    """
    lines = []
    for i, rule in enumerate(refined_rubric_list):
        rule_type = "Hard Rule" if rule.get("type") == "hard" else "Principle"
        lines.append(f"{i+1}. {rule.get('rubric')} [{rule_type}]")
    return "\n".join(lines)

def generate_answers():
    # Load environment variables from .env
    load_dotenv(os.path.join(POINT_RUBRIC_DIR, "..", ".env"))

    # Initialize the client
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI-KEY")
    org_id = os.getenv("OPENAI_ORG_ID")

    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment or .env file.")
        return

    client = OpenAI(
        api_key=api_key,
        organization=org_id,
        timeout=300.0
    )

    # Load questions
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Load prompt templates
    gold_template = load_text(PROMPT_GOLD)
    bad_template = load_text(PROMPT_BAD)
    partial_template = load_text(PROMPT_PARTIAL)
    reason_template = load_text(PROMPT_REASON)
    plain_template = load_text(PROMPT_PLAIN)

    # Define the tasks for each question
    tasks_config = [
        {"id": 1, "model": "gpt-4o", "template": gold_template, "use_rubric": True, "temperature": 0.7},
        {"id": 2, "model": "gpt-4o", "template": bad_template, "use_rubric": True, "temperature": 0.7},
        {"id": 3, "model": "gpt-4o", "template": partial_template, "use_rubric": True, "temperature": 0.7},
        {"id": 4, "model": "gpt-4.1-nano", "template": plain_template, "use_rubric": False, "temperature": 1.3}
    ]

    results = [None] * len(data)
    print(f"Starting answer generation for {len(data)} questions with {MAX_WORKERS} workers...")

    def process_question(index, item):
        question_id = item["question_id"]
        question_text = item["question"]
        refined_rubric = item["refined_rubric"]
        rubric_str = format_rubric_string(refined_rubric)

        print(f"[{index+1}/{len(data)}] Generating 4 answers for question ID: {question_id}...")

        answers = []
        for task in tasks_config:
            try:
                # Prepare the prompt
                prompt = task["template"].replace("{instruction}", question_text)
                if task["use_rubric"]:
                    prompt = prompt.replace("{rubric}", rubric_str)

                # API call arguments
                api_kwargs = {
                    "model": task["model"],
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": task["temperature"]
                }

                response = client.chat.completions.create(**api_kwargs)
                full_answer_text = response.choices[0].message.content.strip()

                if "### Final Answer:" in full_answer_text:
                    answer_text = full_answer_text.split("### Final Answer:")[-1].strip()
                else:
                    answer_text = full_answer_text

                answers.append({
                    "answer_id": task["id"],
                    "model": task["model"],
                    "answer_text": answer_text
                })

                # Tiny delay per task to avoid instant burst limit within one question
                time.sleep(0.2)

            except Exception as e:
                print(f"    [{question_id}] Error in task {task['id']}: {e}")
                answers.append({
                    "answer_id": task["id"],
                    "model": task["model"],
                    "answer_text": f"ERROR: {str(e)}"
                })

        return {
            "question_id": question_id,
            "answers": answers
        }

    # Parallelize at the question level
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_question, i, item): i for i, item in enumerate(data)}
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                res = future.result()
                results[idx] = res

                # Intermediate save after each question completes
                # We filter out Nones and save the whole list to maintain robustness
                final_results = [r for r in results if r is not None]
                with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                    json.dump(final_results, f, indent=4, ensure_ascii=False)
            except Exception as e:
                print(f"Fatal error processing question index {idx}: {e}")

    print(f"\nGeneration complete. Results saved to {OUTPUT_FILE}")

    print(f"\nGeneration complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    try:
        generate_answers()
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
