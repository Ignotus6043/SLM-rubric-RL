# Small Language Models as Judges for Rubric-Based Reinforcement Learning

This repository accompanies **“Small Language Models as Judges for Rubric-Based Reinforcement Learning.”** It releases the camera-ready paper, the PointRubric benchmark, the exact RaR-Science-Static bank and reference labels, frozen evaluation splits, static judge pipelines, and the reward code used to connect rubric judges to reinforcement learning.

## Why this project

Rubric-based RL can evaluate open-ended answers criterion by criterion, but repeatedly generating judge verdicts is expensive. We compare three readouts from small language models:

- **Generative:** generate a criterion verdict.
- **Yes/No Logprob:** compare the log probabilities of the `Yes` and `No` verbalizers.
- **Probe:** classify a frozen hidden representation with a lightweight supervised head.

Across PointRubric and RaR-Science-Static, the Qwen3-1.7B Probe gives the strongest criterion-level agreement among these methods. As a GRPO reward model on RaR-Science, it improves the policy rubric score from **0.232 to 0.643**, versus **0.594** for an 8B Generative judge, while the 8B baseline uses **10.7×** more cumulative reward-judge time at the comparison checkpoint. The paper also evaluates policy transfer to GPQA-Diamond and judge transfer across rubric domains.

The camera-ready paper is in [`paper/`](paper/Small_Language_Models_as_Judges_for_Rubric_Based_Reinforcement_Learning.pdf).

## Repository map

- `PointRubric/`: benchmark data, prompt templates, fixed split, and benchmark utilities.
- `datasets/`: released static data, provenance, checksums, and frozen question IDs.
- `rubric/data/prepare_data.py`: RaR-Science, RaR-Medicine, and HealthBench conversion to the VERL parquet schema.
- `PointRubric/scripts/benchmark_eval.py`: Generative evaluation on PointRubric.
- `rubric/scripts/bench/`: Yes/No Logprob and Probe evaluation on PointRubric, plus human-audit utilities.
- `rubric/scripts/transfer/`: fixed-response-bank construction, judge scoring, calibration, and transfer analysis.
- `rubric/rubric_rl/`: generative and Probe reward paths used with VERL.
- `SFT/`: PointRubric SFT data preparation, evaluation, and portable LLaMA-Factory configs.
- `configs/paper_rl.json`: the camera-ready paper’s RaR-Science GRPO configuration.
- `REPRODUCIBILITY.md`: setup and end-to-end command guide.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Recreate the canonical PointRubric split:

```bash
python PointRubric/scripts/prepare_bench_fixed_split.py \
  --input PointRubric/data/bench.json \
  --output-dir outputs/pointrubric_split \
  --seed 42 --train-frac 0.4 --dev-frac 0.1 --test-frac 0.5
```

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) before running GPU evaluation, SFT, or RL. Model weights, third-party training frameworks, Probe artifacts, and checkpoints are intentionally not committed; the guide identifies how each is obtained or produced.

Verify the committed dataset hashes, split integrity, reference labels, and aggregate scores with:

```bash
python scripts/verify_release.py
```

## Data and release hygiene

The frozen benchmark files include source and checksum records under `datasets/`. Original project contributions are released under [CC BY 4.0](LICENSE); upstream data, model, and software terms are documented in [`THIRD_PARTY_DATA.md`](THIRD_PARTY_DATA.md). Secrets must be supplied through the shell or an untracked `.env`; `.env.example` contains variable names only.

This release excludes API keys, environment files, private paths, Slurm account details, logs, W&B metadata, caches, model weights, and checkpoints.

## Citation

Please cite the paper using [`CITATION.cff`](CITATION.cff). A BibTeX entry can be exported from the repository’s GitHub citation menu.
