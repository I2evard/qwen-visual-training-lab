from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
        import torch_directml
        from peft import LoraConfig, get_peft_model
        from transformers import LlamaConfig, LlamaForCausalLM
    except Exception as exc:  # pragma: no cover - direct environment probe
        print(f"IMPORT_ERROR: {exc}")
        return 1

    dml = torch_directml.device()
    print(f"DIRECTML_DEVICE={dml}")

    try:
        config = LlamaConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
            rms_norm_eps=1e-5,
        )
        model = LlamaForCausalLM(config)

        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
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

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        input_ids = torch.randint(0, config.vocab_size, (2, 16)).to(dml)
        labels = input_ids.clone()

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"PEFT_DIRECTML_LOSS={loss.item():.6f}")
    except Exception as exc:  # pragma: no cover - direct environment probe
        print(f"PEFT_DIRECTML_ERROR: {type(exc).__name__}: {exc}")
        return 1

    print("PEFT_DIRECTML_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
