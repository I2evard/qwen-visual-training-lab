# Qwen visual training lab

**Windows-native AMD + DirectML lab for Qwen-style visual LoRA experiments.**

This repo explores the most practical Windows fallback path for a Qwen-based
visual reviewer: DirectML device checks, PEFT/LoRA smoke tests, and tiny
seed-trace training runs that help answer a simple question:

**How far can the Windows + AMD path go before Linux or cloud GPUs become the
right answer?**

## Goal

Use this repo to test the most realistic Windows-native fallback:

- **DirectML** for GPU visibility on Windows
- **PEFT/LoRA smoke tests** and a small training-step probe
- seed prompts, evals, and datasets copied from the current local visual-agent work

If this lane fails, the next move is **Linux / cloud GPU** for the real LoRA
training run.

## Public-sharing note

This repo intentionally commits only source code, prompts, and setup scripts.
Generated local artifacts are ignored:

- `data\`
- `evals\`
- `runs\`
- `.venv-directml\`

That keeps local traces, eval outputs, adapters, and environment files out of the
public repo.

## Current hypothesis

The current machine can already do:

- local Ollama inference on the AMD GPU
- visual-review prompt packing
- dataset and eval preparation
- tiny DirectML LoRA training against local seed traces

The current machine cannot yet do:

- full Qwen2.5VL LoRA/SFT training through the standard Windows Python GPU stack

So this repo tests how far **DirectML** can go as a Windows-native bridge before
the work needs Linux or cloud GPU capacity.

## Quick start

```powershell
git clone https://github.com/I2evard/qwen-visual-training-lab.git
Set-Location .\qwen-visual-training-lab
.\scripts\setup-directml-env.ps1
.\.venv-directml\Scripts\python.exe .\scripts\smoke_test_directml.py
.\.venv-directml\Scripts\python.exe .\scripts\smoke_test_peft_directml.py
.\.venv-directml\Scripts\python.exe .\scripts\train_seed_directml_lora.py
.\.venv-directml\Scripts\python.exe .\scripts\summarize_training_runs.py
```

To train on local seed traces, first point the sync script at a local source
folder that contains `local-agent-training\dataset.jsonl`,
`local-agent-training\dataset-report.json`, and `local-agent-evals.json`:

```powershell
.\scripts\sync_seed_artifacts.ps1 -SourceRoot "C:\path\to\source"
```

## Repo layout

- `prompts\` - compact prompt/context assets for the visual reviewer
- `scripts\setup-directml-env.ps1` - creates a clean DirectML-focused venv
- `scripts\smoke_test_directml.py` - basic DirectML device + backward-pass probe
- `scripts\smoke_test_peft_directml.py` - tiny `transformers` + `peft` LoRA probe on DirectML
- `scripts\train_seed_directml_lora.py` - first real local seed-trace LoRA training run on DirectML
- `scripts\summarize_training_runs.py` - summarizes ignored local `runs\*\metrics.json` files
- `scripts\sync_seed_artifacts.ps1` - copies current seed dataset/eval artifacts
- `requirements-directml.txt` - base experiment dependencies

Generated `data\`, `evals\`, and `runs\` artifacts are intentionally local-only
and ignored by git. Re-run `sync_seed_artifacts.ps1` to refresh them from your
own local workspace.

## Dataset and eval discipline

The sync script copies two dataset lanes when they exist:

- `data\seed\local-agent-training-dataset.jsonl` - leak-free training subset
  after eval-overlap exclusions. Use this for honest regression checks.
- `data\seed\local-agent-training-dataset-full.jsonl` - fuller local archive.
  Use this only as a DirectML stress/overfit probe because it can include rows
  that overlap the eval set.

That split is deliberate: a bigger run can prove the Windows GPU path survives
more steps, while the smaller leak-free set preserves honest eval discipline.

## Current findings

- `setup-directml-env.ps1` succeeds
- `smoke_test_directml.py` succeeds (`DIRECTML_SMOKE_OK`)
- `peft` import needed `transformers<5`; the 5.x line broke the import stack in this experiment
- The first GPT-2-style PEFT probe failed during `loss.backward()` with:
  - `RuntimeError: The GPU device instance has been suspended`
- The failure was narrowed to the GPT-2 `Conv1D` LoRA target path, not DirectML overall
- `smoke_test_peft_directml.py` now uses a tiny Llama/Qwen-style model with standard linear LoRA targets and succeeds (`PEFT_DIRECTML_OK`)

Interpretation:

- **DirectML is alive enough to be worth probing further**
- **A tiny `transformers` + `peft` LoRA training step can run on DirectML when the target modules are standard linear projections**
- The next risk is scaling from this tiny probe to a real Qwen-style visual model without hitting unsupported DirectML operators or memory limits

## Training status

`train_seed_directml_lora.py` is intentionally a bootstrap trainer, not full
Qwen2.5VL fine-tuning. It trains a tiny Llama/Qwen-style causal LM with LoRA on
the local seed traces, saves metrics, and writes the adapter under ignored
`runs\`.

Current leak-free baseline run:

- command: `.\.venv-directml\Scripts\python.exe .\scripts\train_seed_directml_lora.py --epochs 1 --max-seq-len 96`
- train examples: `1`
- eval examples: `1`
- initial eval loss: `4.251912`
- final eval loss: `4.239905`
- eval loss improvement: `0.28%`
- result: `SEED_DIRECTML_LORA_TRAINING_OK`

First fuller DirectML archive probe:

- command:
  ```powershell
  .\.venv-directml\Scripts\python.exe .\scripts\train_seed_directml_lora.py `
    --dataset .\data\seed\local-agent-training-dataset-full.jsonl `
    --epochs 3 `
    --max-seq-len 128 `
    --hidden-size 96 `
    --intermediate-size 192 `
    --num-hidden-layers 3 `
    --lora-rank 8 `
    --lora-alpha 16
  ```
- train examples: `6`
- eval examples: `2`
- initial eval loss: `4.438062`
- final eval loss: `4.011981`
- eval loss improvement: `9.60%`
- result: `SEED_DIRECTML_LORA_TRAINING_OK`

Summarize local runs:

```powershell
.\.venv-directml\Scripts\python.exe .\scripts\summarize_training_runs.py --output .\runs\training-summary.json
```

## What success looks like

1. DirectML device creation works
2. Tensor ops work on that device
3. A tiny backward pass works
4. We can then attempt a tiny transformer or PEFT probe

## What failure means

If DirectML cannot survive even the tiny backward-pass probe, stop forcing the
Windows lane and move the true training run to Linux or cloud.
