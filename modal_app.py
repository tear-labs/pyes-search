"""Run pyes-search on a Modal GPU.

    modal run modal_app.py                                  # tests (CPU)
    modal run modal_app.py --action eval --args "--model qwen3.5-9b --tasks data/coupled_heldout.jsonl --method bp --out results/9b-bp1"
"""

from __future__ import annotations

import shlex
import subprocess

import modal

ROOT = "/root/pyes-search"
app = modal.App("pyes-search")
weights = modal.Volume.from_name("pyes-search-hf", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.0", "torchvision", "transformers==5.17.0",
                 "flash-linear-attention==0.5.2", "fla-core==0.5.2", "pillow", "pytest>=8.3")
    .env({"PYTHONPATH": ROOT, "HF_HOME": "/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(".", ROOT, copy=False, ignore=[".git", "results", "__pycache__"])
)


@app.function(image=image, cpu=4, memory=16384, timeout=1800)
def tests() -> str:
    run = subprocess.run(["python", "-m", "pytest", "-q", "tests"], cwd=ROOT,
                         capture_output=True, text=True)
    return run.stdout[-4000:] + run.stderr[-2000:]


@app.function(image=image, gpu="B200", cpu=8, memory=131072, timeout=12 * 3600,
              volumes={"/hf": weights})
def evaluate(args: list[str]) -> str:
    run = subprocess.run(["python", "-m", "pyes_search.evaluate", *args], cwd=ROOT,
                         capture_output=True, text=True)
    if run.returncode:
        raise RuntimeError(run.stderr[-4000:])
    return run.stdout.strip().splitlines()[-1]


@app.local_entrypoint()
def main(action: str = "tests", args: str = "") -> None:
    if action == "tests":
        print(tests.remote())
    elif action == "eval":
        print(evaluate.remote(shlex.split(args)))
    else:
        raise ValueError(action)
