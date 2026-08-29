"""
Judge client with a direct OpenAI SDK path for OpenAI models and LiteLLM as a
fallback for local / non-OpenAI backends.

Environment variables:
    JUDGE_MODEL: Model name
        - OpenAI: "openai/gpt-5.4-mini" or bare "gpt-5.4-mini"
        - Local / LiteLLM: e.g. "hosted_vllm/Qwen/Qwen3-8B"
    JUDGE_API_BASE: Optional API base URL (for local OpenAI-compatible backends)
    JUDGE_API_KEY: Optional API key
    JUDGE_MAX_CONCURRENT: Max concurrent calls (default: 256)
    JUDGE_TIMEOUT: Timeout in seconds (default: 600)
    JUDGE_RETRY_ATTEMPTS: Number of attempts per judge call (default: 1)
    JUDGE_RETRY_BACKOFF_SECONDS: Initial retry backoff in seconds (default: 2)
"""

import os
import asyncio
import json
import threading
import time
import weakref
from pathlib import Path
from typing import Optional

import litellm
from openai import AsyncOpenAI

# Configure litellm to drop unsupported parameters instead of raising errors
litellm.drop_params = True
litellm.set_verbose = False

# Per-event-loop concurrency control for litellm async calls
_JUDGE_SEMAPHORES = weakref.WeakKeyDictionary()
_USAGE_LOG_LOCK = threading.Lock()


def _get_judge_semaphore() -> asyncio.Semaphore:
    """Return a per-event-loop semaphore limiting concurrent judge calls."""
    loop = asyncio.get_running_loop()
    sem = _JUDGE_SEMAPHORES.get(loop)
    if sem is None:
        max_concurrent = int(os.environ.get("JUDGE_MAX_CONCURRENT", "256"))
        sem = asyncio.Semaphore(max_concurrent)
        _JUDGE_SEMAPHORES[loop] = sem
    return sem


def _get_judge_config() -> dict:
    """Get judge configuration from environment variables."""
    config = {
        "model": os.environ.get("JUDGE_MODEL", "gpt-4.1-mini"),
        "api_base": os.environ.get("JUDGE_API_BASE"),
        "api_key": os.environ.get("JUDGE_API_KEY"),
        "timeout": float(os.environ.get("JUDGE_TIMEOUT", "600")),
        "temperature": 0,
        "max_tokens": int(os.environ.get("JUDGE_MAX_TOKENS", "2048")),
    }
    return config


def _judge_retry_attempts() -> int:
    try:
        return max(1, int(os.environ.get("JUDGE_RETRY_ATTEMPTS", "1")))
    except ValueError:
        return 1


def _judge_retry_backoff_seconds() -> float:
    try:
        return max(0.0, float(os.environ.get("JUDGE_RETRY_BACKOFF_SECONDS", "2")))
    except ValueError:
        return 2.0


def _disable_qwen3_thinking() -> bool:
    value = os.environ.get("JUDGE_DISABLE_THINKING", "false")
    return value.lower() in {"1", "true", "yes", "on"}


def _qwen3_chat_template_kwargs(model: str) -> Optional[dict]:
    if not _disable_qwen3_thinking():
        return None
    if "qwen3" not in model.lower():
        return None
    return {"enable_thinking": False}


def _thinking_token_budget() -> Optional[int]:
    value = os.environ.get("JUDGE_THINKING_TOKEN_BUDGET", "").strip()
    if not value:
        return None
    try:
        budget = int(value)
    except ValueError:
        return None
    return budget if budget > 0 else None


def _extra_body_for_model(model: str) -> Optional[dict]:
    extra_body: dict = {}
    chat_template_kwargs = _qwen3_chat_template_kwargs(model)
    if chat_template_kwargs is not None:
        extra_body["chat_template_kwargs"] = chat_template_kwargs
    thinking_token_budget = _thinking_token_budget()
    if thinking_token_budget is not None:
        extra_body["thinking_token_budget"] = thinking_token_budget
    return extra_body or None


def _usage_to_dict(usage: object) -> Optional[dict]:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return dict(usage)
    result = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, key, None)
        if value is not None:
            result[key] = value
    return result or None


def _append_usage_log(*, backend: str, model: str, messages: list[dict[str, str]], usage: object) -> None:
    log_path = os.environ.get("JUDGE_USAGE_LOG", "").strip()
    if not log_path:
        return
    usage_dict = _usage_to_dict(usage)
    if usage_dict is None:
        return

    prompt_chars = sum(len(message.get("content", "")) for message in messages)
    record = {
        "ts": time.time(),
        "backend": backend,
        "model": model,
        "prompt_chars": prompt_chars,
        "num_messages": len(messages),
        "usage": usage_dict,
    }
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _USAGE_LOG_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _normalize_openai_model_name(model: str) -> str:
    """Return the raw OpenAI model id without provider prefix."""
    if model.startswith("openai/"):
        return model.split("/", 1)[1]
    return model


def _uses_reasoning_chat_params(model: str) -> bool:
    """Return whether an OpenAI chat model uses reasoning-era parameters."""
    raw_model = _normalize_openai_model_name(model).lower()
    return raw_model.startswith(("gpt-5", "o1", "o3", "o4"))


def _reasoning_effort_for_model(model: str) -> Optional[str]:
    """Choose a cheap/default reasoning effort for judge-style calls."""
    env_value = os.environ.get("JUDGE_REASONING_EFFORT", "").strip()
    if env_value:
        return env_value
    raw_model = _normalize_openai_model_name(model).lower()
    if raw_model.startswith("gpt-5"):
        return "minimal"
    return None


def _should_use_openai_sdk(config: dict) -> bool:
    """
    Route explicit OpenAI models, and bare GPT/o-series names without a custom
    API base, through the official OpenAI SDK.
    """
    model = config["model"]
    if model.startswith("openai/"):
        return True
    if config["api_base"]:
        return False
    return model.startswith(("gpt-", "o1", "o3", "o4"))


async def _call_openai_judge(config: dict, messages: list[dict[str, str]]) -> str:
    api_key = config["api_key"] or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set for OpenAI judge model "
            f"{config['model']!r}. Load a .env file or export OPENAI_API_KEY."
        )
    client = AsyncOpenAI(
        api_key=api_key,
        organization=os.environ.get("OPENAI_ORG_ID") or None,
        timeout=config["timeout"],
    )
    raw_model = _normalize_openai_model_name(config["model"])
    request_kwargs = {
        "model": raw_model,
        "messages": messages,
    }
    if _uses_reasoning_chat_params(config["model"]):
        request_kwargs["max_completion_tokens"] = config["max_tokens"]
        reasoning_effort = _reasoning_effort_for_model(config["model"])
        if reasoning_effort:
            request_kwargs["reasoning_effort"] = reasoning_effort
    else:
        request_kwargs["temperature"] = config["temperature"]
        request_kwargs["max_tokens"] = config["max_tokens"]

    response = await client.chat.completions.create(**request_kwargs)
    _append_usage_log(
        backend="openai_sdk",
        model=_normalize_openai_model_name(config["model"]),
        messages=messages,
        usage=getattr(response, "usage", None),
    )
    content = response.choices[0].message.content
    return content or ""


async def call_judge(prompt: str, system_prompt: Optional[str] = None) -> str:
    """
    Call the judge model to evaluate a response.

    Args:
        prompt: The user prompt to send to the judge
        system_prompt: Optional system prompt

    Returns:
        Judge response text, or empty string on failure
    """
    config = _get_judge_config()

    # Prepare messages
    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    else:
        messages = [{"role": "user", "content": prompt}]

    attempts = _judge_retry_attempts()
    backoff_seconds = _judge_retry_backoff_seconds()

    for attempt_idx in range(1, attempts + 1):
        # Guard concurrent calls with a global semaphore
        try:
            semaphore = _get_judge_semaphore()
            async with semaphore:
                if _should_use_openai_sdk(config):
                    return await _call_openai_judge(config, messages)

                litellm_kwargs = {
                    "model": config["model"],
                    "messages": messages,
                    "temperature": config["temperature"],
                    "max_tokens": config["max_tokens"],
                    "timeout": config["timeout"],
                }
                if config["api_base"]:
                    litellm_kwargs["api_base"] = config["api_base"]
                if config["api_key"]:
                    litellm_kwargs["api_key"] = config["api_key"]
                extra_body = _extra_body_for_model(config["model"])
                if extra_body is not None:
                    litellm_kwargs["extra_body"] = extra_body
                response = await litellm.acompletion(**litellm_kwargs)
                _append_usage_log(
                    backend="litellm",
                    model=config["model"],
                    messages=messages,
                    usage=getattr(response, "usage", None),
                )
            return response.choices[0].message.content
        except Exception as e:
            if attempt_idx >= attempts:
                print(f"Error calling judge after {attempts} attempt(s): {e}")
                return ""
            sleep_seconds = min(backoff_seconds * (2 ** (attempt_idx - 1)), 30.0)
            print(f"Error calling judge on attempt {attempt_idx}/{attempts}: {e}. Retrying in {sleep_seconds:.1f}s")
            if sleep_seconds > 0:
                await asyncio.sleep(sleep_seconds)

    return ""


async def call_judge_batch(prompts: list, system_prompt: Optional[str] = None) -> list:
    """
    Call the judge model for multiple prompts concurrently.

    Args:
        prompts: List of user prompts to send to the judge
        system_prompt: Optional system prompt

    Returns:
        List of judge response texts (empty string for failed calls)
    """
    tasks = [call_judge(prompt, system_prompt) for prompt in prompts]
    return await asyncio.gather(*tasks)


if __name__ == "__main__":
    # Simple test
    async def test_judge():
        import sys
        if len(sys.argv) > 1:
            os.environ["JUDGE_MODEL"] = sys.argv[1]
        if len(sys.argv) > 2:
            os.environ["JUDGE_API_BASE"] = sys.argv[2]

        print(f"Using judge model: {os.environ.get('JUDGE_MODEL', 'gpt-4.1-mini')}")

        test_prompt = "What is 2 + 2? Provide your answer in \\boxed{} format."
        response = await call_judge(test_prompt)
        print(f"Response: {response}")

    asyncio.run(test_judge())
