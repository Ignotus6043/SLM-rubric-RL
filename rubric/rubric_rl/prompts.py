"""
Prompt templates for rubric-based reward scoring.

This module provides prompt templates for two scoring strategies:
1. Explicit: Score one response against all rubric criteria in a single call
2. Implicit: Score all rubric items in a single call (Likert 1-10)
"""

EXPLICIT_PROMPT_TEMPLATE = """
You are evaluating whether an AI response satisfies a specific rubric criterion.

[Question]
{question}

[Criterion to Evaluate]
Title: {title}
Description: {description}

[AI Response]
{response}

Evaluate whether the AI response satisfies this criterion.
Output exactly one final line and no explanation:
- 1 if the criterion is satisfied
- 0 if the criterion is NOT satisfied

Your entire response must be exactly one of:
\\boxed{{0}}
\\boxed{{1}}
""".strip()


EXPLICIT_JOINT_PROMPT_TEMPLATE = """
You are an impartial rubric grader.

Instruction:
{question}

Candidate Response:
{response}

Rubric Rules ({num_rules} total):
{rubric_rules}

Task:
For each rubric rule, decide whether the response satisfies it.

Output format requirement (strict):
- Return exactly {num_rules} lines.
- Each line must follow this format exactly:
  Rule i: <short justification>. Verdict: Yes/No
- Use i = 1..{num_rules} in order.
- Keep each justification brief (1 sentence).
- Verdict must be exactly `Yes` or `No`.

Example for 6 rules:
Rule 1: [Justification for rule 1]. Verdict: Yes
Rule 2: [Justification for rule 2]. Verdict: No
...
Rule 6: [Justification for rule 6]. Verdict: Yes

Do not output JSON, markdown, extra headings, or any text outside those rule lines.
""".strip()


IMPLICIT_PROMPT_TEMPLATE = """
You are evaluating the overall quality of an AI response against a rubric.

[Question]
{question}

[Rubric]
{rubric_formatted}

[AI Response]
{response}

Evaluate the overall quality of the AI response holistically, considering all rubric criteria together.
Provide a single score on a Likert scale from 1 to 10 (1 = very poor, 10 = excellent).

Final answer in \\boxed{{score}}.
""".strip()


def format_rubric_for_implicit(rubric):
    """
    Format rubric items for the implicit scoring strategy.

    Args:
        rubric: List of dicts with keys: title, description (optional: weight)

    Returns:
        Formatted string representation of the rubric
    """
    lines = []
    for item in rubric:
        lines.append(f"- {item['title']}: {item['description']}")
    return "\n".join(lines)


def build_explicit_prompt(criterion, question, response):
    """
    Build a prompt for evaluating a single rubric criterion.

    Args:
        criterion: Dict with keys: title, description (optional: weight)
        question: The original question
        response: The AI response to evaluate

    Returns:
        Formatted prompt string
    """
    return EXPLICIT_PROMPT_TEMPLATE.format(
        question=question,
        title=criterion["title"],
        description=criterion["description"],
        response=response,
    )


def format_rubric_for_explicit_joint(rubric):
    """
    Format rubric items for the explicit joint scoring strategy.

    Each line maps to a canonical rule id that will be parsed back from the
    judge output, mirroring the strict format used in Bench.
    """
    lines = []
    for idx, item in enumerate(rubric, start=1):
        title = item.get("title", "").strip()
        description = item.get("description", "").strip()
        if title and description:
            text = f"{title}: {description}"
        else:
            text = title or description
        lines.append(f"{idx}. {text}")
    return "\n".join(lines)


def build_explicit_joint_prompt(question, rubric, response):
    """
    Build a prompt for evaluating all rubric criteria for a single response.

    Args:
        question: The original question
        rubric: List of rubric criteria dicts
        response: The AI response to evaluate

    Returns:
        Formatted prompt string
    """
    rubric_rules = format_rubric_for_explicit_joint(rubric)
    return EXPLICIT_JOINT_PROMPT_TEMPLATE.format(
        question=question,
        rubric_rules=rubric_rules,
        num_rules=len(rubric),
        response=response,
    )


def build_implicit_prompt(question, rubric, response):
    """
    Build a prompt for evaluating all rubric criteria in one call.

    Args:
        question: The original question
        rubric: List of rubric criteria dicts
        response: The AI response to evaluate

    Returns:
        Formatted prompt string
    """
    rubric_formatted = format_rubric_for_implicit(rubric)
    return IMPLICIT_PROMPT_TEMPLATE.format(
        question=question,
        rubric_formatted=rubric_formatted,
        response=response,
    )
