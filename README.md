# Qwen visual training lab

Experimental repo for getting a **Windows + AMD** training path working for a
Qwen-based visual reviewer.

## Goal

Use this repo to test the most realistic Windows-native fallback:

- **DirectML** for GPU visibility on Windows
- a small **training-step smoke test**
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

The current machine cannot yet do:

- real PyTorch LoRA/SFT training on a GPU-visible backend through the standard
  Windows Python stack

So this repo tests whether **DirectML** can become that missing Windows-native
bridge.

## Quick start

```powershell
git clone https://github.com/I2evard/qwen-visual-training-lab.git
Set-Location .\qwen-visual-training-lab
.\scripts\setup-directml-env.ps1
.\.venv-directml\Scripts\python.exe .\scripts\smoke_test_directml.py
.\.venv-directml\Scripts\python.exe .\scripts\smoke_test_peft_directml.py
.\.venv-directml\Scripts\python.exe .\scripts\train_seed_directml_lora.py
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
- `scripts\sync_seed_artifacts.ps1` - copies current seed dataset/eval artifacts
- `requirements-directml.txt` - base experiment dependencies

Generated `data\` and `evals\` artifacts are intentionally local-only and ignored
by git. Re-run `sync_seed_artifacts.ps1` to refresh them from your own local
workspace.

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

First verified run:

- command: `.\.venv-directml\Scripts\python.exe .\scripts\train_seed_directml_lora.py --epochs 1 --max-seq-len 96`
- train examples: `13`
- eval examples: `4`
- initial eval loss: `4.298570`
- final eval loss: `4.132506`
- result: `SEED_DIRECTML_LORA_TRAINING_OK`

## What success looks like

1. DirectML device creation works
2. Tensor ops work on that device
3. A tiny backward pass works
4. We can then attempt a tiny transformer or PEFT probe

## What failure means

If DirectML cannot survive even the tiny backward-pass probe, stop forcing the
Windows lane and move the true training run to Linux or cloud.
