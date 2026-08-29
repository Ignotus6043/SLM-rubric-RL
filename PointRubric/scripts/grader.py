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
REFINED_RUBRICS_FILE = os.path.join(DATA_DIR, "retrieved_questions_refined_rubrics.json")
GENERATED_ANSWERS_FILE = os.path.join(DATA_DIR, "generated_answers.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "bench.json")
GRADER_PROMPT_FILE = os.path.join(POINT_RUBRIC_DIR, "rubric_prompts", "grader.txt")
MODEL = "gpt-4o"
MAX_WORKERS = 8 # Number of parallel grading tasks

def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def format_rubric_rules(refined_rubric_list):
    """
    Converts the refined rubric list into a string format for the grader prompt.
    """
    lines = []
    for rule in refined_rubric_list:
        rule_type = "Hard Rule" if rule.get("type") == "hard" else "Soft Rule"
        lines.append(f"{rule['rule_id']}. {rule['rubric']} [{rule_type}]")
    return "\n".join(lines)

def run_grader():
    # Load environment variables
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

    # Load data
    if not os.path.exists(REFINED_RUBRICS_FILE) or not os.path.exists(GENERATED_ANSWERS_FILE):
        print("Error: Required input files not found.")
        return

    with open(REFINED_RUBRICS_FILE, "r", encoding="utf-8") as f:
        refined_data = json.load(f)

    with open(GENERATED_ANSWERS_FILE, "r", encoding="utf-8") as f:
        answers_data = json.load(f)

    grader_template = load_text(GRADER_PROMPT_FILE)
    answers_lookup = {item["question_id"]: item["answers"] for item in answers_data}

    # Initialize bench_data structure
    bench_data = []
    bench_lookup = {}
    for item in refined_data:
        q_id = item["question_id"]
        bench_lookup[q_id] = {
            "question_id": q_id,
            "question": item["question"],
            "original_rubric": item["original_rubric"],
            "refined_rubric": item["refined_rubric"],
            "answers": [ans.copy() for ans in answers_lookup.get(q_id, [])]
        }
        bench_data.append(bench_lookup[q_id])

    print(f"Starting parallel grading for {len(refined_data)} questions using {MODEL} with {MAX_WORKERS} workers...")

    def grade_single_answer(q_id, q_text, ans, ref_rubric, rubric_rules_str):
        print(f"  [{q_id}] Grading answer {ans['answer_id']}...")
        try:
            prompt = grader_template.replace("{instruction}", q_text)
            prompt = prompt.replace("{response}", ans["answer_text"])
            prompt = prompt.replace("{rubric_rules}", rubric_rules_str)

            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are an impartial and meticulous grader."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )

            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[len("```json"):].strip()
            if content.endswith("```"):
                content = content[:-3].strip()

            grade_json = json.loads(content)
            rule_type_map = {str(r['rule_id']): r['type'] for r in ref_rubric}

            total_score = 0
            for rule in grade_json:
                if rule.get("verdict") == "yes":
                    actual_type = rule_type_map.get(str(rule.get("rule_id")))
                    total_score += 3 if actual_type == "hard" else 1

            return {
                "grade": grade_json,
                "total_score": total_score
            }
        except Exception as e:
            print(f"    [{q_id}] Error grading answer {ans['answer_id']}: {e}")
            return None

    # Flatten all (question, answer) pairs for parallelization
    grading_tasks = []
    for q_item in bench_data:
        q_id = q_item["question_id"]
        q_text = q_item["question"]
        ref_rubric = q_item["refined_rubric"]
        rubric_rules_str = format_rubric_rules(ref_rubric)
        for ans in q_item["answers"]:
            grading_tasks.append((q_id, q_text, ans, ref_rubric, rubric_rules_str))

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {
            executor.submit(grade_single_answer, *task): task
            for task in grading_tasks
        }
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            res = future.result()
            if res:
                # Find the answer object in our bench_data and update it
                q_id, _, original_ans, _, _ = task
                original_ans.update(res)

    # Save results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(bench_data, f, indent=4, ensure_ascii=False)

    print(f"\nGrading complete. Compiled results saved to {OUTPUT_FILE}")

    print(f"\nGrading complete. Compiled results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    try:
        run_grader()
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
