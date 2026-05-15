from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    dataset_path: str
    output_dir: str
    seed: int
    eval_fraction: float
    epochs: int
    learning_rate: float
    max_seq_len: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    lora_rank: int
    lora_alpha: int


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train a tiny Qwen/Llama-style LoRA adapter on the local seed traces through DirectML."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=repo_root / "data" / "seed" / "local-agent-training-dataset.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "runs" / f"seed-directml-lora-{time.strftime('%Y%m%d-%H%M%S')}",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--intermediate-size", type=int, default=128)
    parser.add_argument("--num-hidden-layers", type=int, default=2)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    return parser.parse_args()


def format_example(record: dict[str, object]) -> str:
    task = str(record.get("task") or "").strip()
    response = str(record.get("response") or "").strip()
    return f"<user>\n{task}\n</user>\n<assistant>\n{response}\n</assistant>"


def load_texts(dataset_path: Path) -> list[str]:
    if not dataset_path.exists():
        raise SystemExit(
            f"Dataset not found: {dataset_path}. Run scripts\\sync_seed_artifacts.ps1 first."
        )

    texts: list[str] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            text = format_example(record)
            if text:
                texts.append(text)

    if len(texts) < 2:
        raise SystemExit("Need at least 2 usable examples to create train/eval splits.")

    return texts


def build_vocab(texts: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    chars = sorted(set("".join(texts)))
    special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
    tokens = special_tokens + chars
    token_to_id = {token: index for index, token in enumerate(tokens)}
    id_to_token = {index: token for token, index in token_to_id.items()}
    return token_to_id, id_to_token


def encode_text(text: str, token_to_id: dict[str, int], max_seq_len: int) -> list[int]:
    bos = token_to_id["<bos>"]
    eos = token_to_id["<eos>"]
    unk = token_to_id["<unk>"]
    encoded = [bos] + [token_to_id.get(char, unk) for char in text] + [eos]
    return encoded[:max_seq_len]


def make_batch(
    examples: list[list[int]],
    *,
    pad_token_id: int,
    device: object,
    torch_module: object,
) -> tuple[object, object]:
    max_len = max(len(example) for example in examples)
    input_rows: list[list[int]] = []
    label_rows: list[list[int]] = []

    for example in examples:
        padded = example + [pad_token_id] * (max_len - len(example))
        labels = [token if token != pad_token_id else -100 for token in padded]
        input_rows.append(padded)
        label_rows.append(labels)

    input_ids = torch_module.tensor(input_rows, dtype=torch_module.long).to(device)
    labels = torch_module.tensor(label_rows, dtype=torch_module.long).to(device)
    return input_ids, labels


def evaluate(
    *,
    model: object,
    encoded_eval: list[list[int]],
    pad_token_id: int,
    device: object,
    torch_module: object,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch_module.no_grad():
        for encoded in encoded_eval:
            input_ids, labels = make_batch(
                [encoded],
                pad_token_id=pad_token_id,
                device=device,
                torch_module=torch_module,
            )
            outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
            losses.append(float(outputs.loss.detach().cpu()))
    model.train()
    return sum(losses) / len(losses)


def main() -> int:
    args = parse_args()

    try:
        import torch
        import torch_directml
        from peft import LoraConfig, get_peft_model
        from transformers import LlamaConfig, LlamaForCausalLM
    except Exception as exc:
        print(f"IMPORT_ERROR: {exc}")
        return 1

    random.seed(args.seed)
    texts = load_texts(args.dataset)
    random.shuffle(texts)

    eval_count = max(1, math.ceil(len(texts) * args.eval_fraction))
    eval_texts = texts[:eval_count]
    train_texts = texts[eval_count:]
    if not train_texts:
        train_texts = texts[1:]
        eval_texts = texts[:1]

    token_to_id, id_to_token = build_vocab(texts)
    pad_token_id = token_to_id["<pad>"]
    encoded_train = [encode_text(text, token_to_id, args.max_seq_len) for text in train_texts]
    encoded_eval = [encode_text(text, token_to_id, args.max_seq_len) for text in eval_texts]

    dml = torch_directml.device()
    print(f"DIRECTML_DEVICE={dml}")
    print(f"TRAIN_EXAMPLES={len(encoded_train)}")
    print(f"EVAL_EXAMPLES={len(encoded_eval)}")
    print(f"VOCAB_SIZE={len(token_to_id)}")

    model_config = LlamaConfig(
        vocab_size=len(token_to_id),
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_attention_heads,
        max_position_embeddings=args.max_seq_len,
        rms_norm_eps=1e-5,
        pad_token_id=pad_token_id,
        bos_token_id=token_to_id["<bos>"],
        eos_token_id=token_to_id["<eos>"],
    )
    model = LlamaForCausalLM(model_config)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()
    model.to(dml)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    losses: list[dict[str, float | int]] = []
    initial_eval_loss = evaluate(
        model=model,
        encoded_eval=encoded_eval,
        pad_token_id=pad_token_id,
        device=dml,
        torch_module=torch,
    )
    print(f"INITIAL_EVAL_LOSS={initial_eval_loss:.6f}")

    step = 0
    for epoch in range(1, args.epochs + 1):
        random.shuffle(encoded_train)
        for encoded in encoded_train:
            step += 1
            input_ids, labels = make_batch(
                [encoded],
                pad_token_id=pad_token_id,
                device=dml,
                torch_module=torch,
            )
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            losses.append({"step": step, "epoch": epoch, "train_loss": loss_value})
            print(f"step={step} epoch={epoch} train_loss={loss_value:.6f}")

    final_eval_loss = evaluate(
        model=model,
        encoded_eval=encoded_eval,
        pad_token_id=pad_token_id,
        device=dml,
        torch_module=torch,
    )
    print(f"FINAL_EVAL_LOSS={final_eval_loss:.6f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.to("cpu")
    model.save_pretrained(args.output_dir / "adapter")

    config = TrainConfig(
        dataset_path=str(args.dataset),
        output_dir=str(args.output_dir),
        seed=args.seed,
        eval_fraction=args.eval_fraction,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_seq_len=args.max_seq_len,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )
    metrics = {
        "status": "completed",
        "directml_device": str(dml),
        "train_examples": len(encoded_train),
        "eval_examples": len(encoded_eval),
        "vocab_size": len(token_to_id),
        "initial_eval_loss": initial_eval_loss,
        "final_eval_loss": final_eval_loss,
        "losses": losses,
        "config": asdict(config),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "vocab.json").write_text(
        json.dumps(token_to_id, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"TRAINING_RUN_DIR={args.output_dir}")
    print("SEED_DIRECTML_LORA_TRAINING_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
