#!/usr/bin/env python3
"""HTTP reward server for frozen representation probes.

The server exposes the same per-criterion reward shape as the generated-judge
reward function, but obtains each criterion score from a saved linear probe over
frozen LM hidden states. It is intended for RL runs where the reward model runs
on a separate GPU from the actor/rollout worker.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_PROBE_HELPER_PATH = REPO_ROOT / "scripts" / "transfer" / "score_rarscience_response_bank_hf_probe.py"
_PROBE_HELPER_SPEC = importlib.util.spec_from_file_location("rarscience_probe_helper", _PROBE_HELPER_PATH)
if _PROBE_HELPER_SPEC is None or _PROBE_HELPER_SPEC.loader is None:
    raise ImportError(f"Could not load probe helper from {_PROBE_HELPER_PATH}")
_PROBE_HELPER = importlib.util.module_from_spec(_PROBE_HELPER_SPEC)
sys.modules[_PROBE_HELPER_SPEC.name] = _PROBE_HELPER
_PROBE_HELPER_SPEC.loader.exec_module(_PROBE_HELPER)

build_prompt = _PROBE_HELPER.build_prompt
build_probe_classifier = _PROBE_HELPER.build_probe_classifier
extract_features = _PROBE_HELPER.extract_features
format_criterion = _PROBE_HELPER.format_criterion
load_tokenizer = _PROBE_HELPER.load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, help="Path to probe_artifact.pt.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8621)
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rubric-format", default="", help="Override artifact rubric format when non-empty.")
    parser.add_argument("--max-rubric-rule-chars", type=int, default=-1)
    parser.add_argument("--score-mode", choices=("probability", "binary"), default="probability")
    parser.add_argument(
        "--calibration-mode",
        choices=("artifact", "none"),
        default="artifact",
        help="Apply calibration metadata stored in the artifact when present.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def load_artifact(path: str | Path) -> dict[str, Any]:
    artifact = torch.load(path, map_location="cpu")
    if not isinstance(artifact, dict):
        raise TypeError(f"Expected dict artifact at {path}")
    if artifact.get("probe_classifier") != "linear":
        raise ValueError(f"Only linear probe artifacts are supported for RL; got {artifact.get('probe_classifier')}")
    for key in ("judge_model", "best_layer", "best_threshold", "state_dict", "scaler", "feature_dim"):
        if key not in artifact:
            raise KeyError(f"Probe artifact missing required key: {key}")
    return artifact


def aggregate_score(rubric: list[dict[str, Any]], values: list[float]) -> tuple[float, list[float]]:
    weights = [abs(float(item.get("weight", 1.0) if item.get("weight", 1.0) is not None else 1.0)) for item in rubric]
    total_positive_weight = sum(weight for weight in weights if weight > 0)
    if total_positive_weight <= 0:
        return 0.0, weights
    score = sum(value * weight for value, weight in zip(values, weights)) / total_positive_weight
    return float(score), weights


def calibration_from_artifact(artifact: dict[str, Any], mode: str) -> dict[str, Any] | None:
    if mode == "none":
        return None
    calibration = artifact.get("calibration")
    if not isinstance(calibration, dict):
        return None
    method = str(calibration.get("method", "")).lower()
    if method not in {"platt", "temperature"}:
        raise ValueError(f"Unsupported probe calibration method: {method}")
    return calibration


class ProbeRewardScorer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.artifact_path = Path(args.artifact)
        self.artifact = load_artifact(self.artifact_path)
        self.judge_model = str(self.artifact["judge_model"])
        self.layer = int(self.artifact["best_layer"])
        self.raw_threshold = float(self.artifact["best_threshold"])
        self.calibration = calibration_from_artifact(self.artifact, args.calibration_mode)
        self.threshold = (
            float(self.calibration.get("threshold", 0.5))
            if self.calibration is not None
            else self.raw_threshold
        )
        self.feature_dim = int(self.artifact["feature_dim"])
        self.rubric_format = args.rubric_format or str(self.artifact.get("rubric_format", "canonical") or "canonical")
        artifact_max_chars = int(self.artifact.get("max_rubric_rule_chars", 0) or 0)
        self.max_rubric_rule_chars = artifact_max_chars if args.max_rubric_rule_chars < 0 else args.max_rubric_rule_chars
        self.lock = threading.Lock()

        self.tokenizer = load_tokenizer(self.judge_model, args.trust_remote_code)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.judge_model,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=args.trust_remote_code,
        )
        self.model.eval()

        scaler = self.artifact["scaler"]
        self.mean = scaler["mean"].float()
        self.std = scaler["std"].float().clamp_min(1e-6)
        self.probe = build_probe_classifier(self.feature_dim, "linear", 0, 0.0)
        self.probe.load_state_dict(self.artifact["state_dict"])
        self.probe.eval()

    def score(self, payload: dict[str, Any]) -> dict[str, Any]:
        question = str(payload.get("question", "") or "")
        response = str(payload.get("response", "") or "")
        rubric = payload.get("rubric") or []
        if not isinstance(rubric, list) or not rubric:
            return {
                "score": 0.0,
                "criterion_scores": [],
                "criterion_weights": [],
                "criterion_parse_ok": [],
                "num_parse_failures": 0,
                "parse_source": "probe_server_empty_rubric",
                "raw_parse_success": True,
                "fallback_used": False,
            }

        items = []
        criterion_text = []
        for rule_index, rule in enumerate(rubric):
            if not isinstance(rule, dict):
                rule = {"description": str(rule)}
            criterion = format_criterion(rule, self.rubric_format, self.max_rubric_rule_chars)
            criterion_text.append(criterion)
            items.append(
                {
                    "rule_index": rule_index,
                    "prompt": build_prompt(question, response, criterion),
                    "criterion": criterion,
                }
            )

        started = time.perf_counter()
        # Serialize model forward passes inside this process. ThreadingHTTPServer
        # still lets concurrent clients queue without corrupting CUDA state.
        with self.lock, torch.inference_mode():
            features = extract_features(
                items,
                self.model,
                self.tokenizer,
                [self.layer],
                self.args.batch_size,
                self.args.max_prompt_length,
            )[self.layer]
            if features.shape[-1] != self.feature_dim:
                raise ValueError(f"Feature dim mismatch: target={features.shape[-1]} artifact={self.feature_dim}")
            x_eval = (features - self.mean) / self.std
            logits = self.probe(x_eval).squeeze(-1)
            raw_probs = torch.sigmoid(logits)
            if self.calibration is not None:
                a = float(self.calibration.get("a", 1.0))
                b = float(self.calibration.get("b", 0.0))
                calibrated_logits = logits * a + b
                probs = torch.sigmoid(calibrated_logits)
            else:
                calibrated_logits = logits
                probs = raw_probs

        yes_probs = [float(value) for value in probs.cpu().tolist()]
        raw_yes_probs = [float(value) for value in raw_probs.cpu().tolist()]
        logits_list = [float(value) for value in logits.cpu().tolist()]
        calibrated_logits_list = [float(value) for value in calibrated_logits.cpu().tolist()]
        verdicts = [1.0 if prob >= self.threshold else 0.0 for prob in yes_probs]
        values = yes_probs if self.args.score_mode == "probability" else verdicts
        score, weights = aggregate_score(rubric, values)
        elapsed = time.perf_counter() - started

        return {
            "score": score,
            "criterion_scores": values,
            "criterion_weights": weights,
            "criterion_parse_ok": [True] * len(rubric),
            "num_parse_failures": 0,
            "raw_num_parse_failures": 0,
            "raw_parse_success": True,
            "fallback_used": False,
            "parse_source": "hf_representation_probe_server",
            "judge_attempts": len(rubric),
            "raw_judge_response": "",
            "raw_judge_responses": [],
            "criterion_yes_probs": yes_probs,
            "criterion_probe_logits": logits_list,
            "criterion_text": criterion_text,
            "criterion_raw_yes_probs": raw_yes_probs,
            "criterion_calibrated_logits": calibrated_logits_list,
            "probe_threshold": self.threshold,
            "probe_raw_threshold": self.raw_threshold,
            "probe_layer": self.layer,
            "probe_artifact": str(self.artifact_path),
            "judge_model": self.judge_model,
            "probe_rubric_format": self.rubric_format,
            "probe_score_mode": self.args.score_mode,
            "probe_calibration": self.calibration or {"method": "none"},
            "probe_elapsed_seconds": elapsed,
        }


def make_handler(scorer: ProbeRewardScorer):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ProbeRewardServer/0.1"

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:  # noqa: N802
            if self.path in {"/health", "/v1/models"}:
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "model": scorer.judge_model,
                        "artifact": str(scorer.artifact_path),
                        "layer": scorer.layer,
                        "threshold": scorer.threshold,
                        "calibration": scorer.calibration or {"method": "none"},
                    },
                )
                return
            self._send_json(404, {"ok": False, "error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path not in {"/score", "/v1/score"}:
                self._send_json(404, {"ok": False, "error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                result = scorer.score(payload)
                self._send_json(200, {"ok": True, "result": result})
            except Exception as exc:  # pragma: no cover - defensive server guard
                self._send_json(500, {"ok": False, "error": repr(exc)})

        def log_message(self, format: str, *args: Any) -> None:
            sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), format % args))

    return Handler


def main() -> int:
    args = parse_args()
    scorer = ProbeRewardScorer(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(scorer))
    print("=======================================", flush=True)
    print("Probe reward server ready", flush=True)
    print(f"Artifact:       {args.artifact}", flush=True)
    print(f"Judge model:    {scorer.judge_model}", flush=True)
    print(f"Layer:          {scorer.layer}", flush=True)
    print(f"Threshold:      {scorer.threshold:.6f}", flush=True)
    print(f"Calibration:    {(scorer.calibration or {'method': 'none'}).get('method')}", flush=True)
    print(f"Rubric format:  {scorer.rubric_format}", flush=True)
    print(f"Score mode:     {args.score_mode}", flush=True)
    print(f"Address:        http://{args.host}:{args.port}", flush=True)
    print("=======================================", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
