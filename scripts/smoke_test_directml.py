from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
        import torch_directml
    except Exception as exc:  # pragma: no cover - direct environment probe
        print(f"IMPORT_ERROR: {exc}")
        return 1

    print(f"TORCH_VERSION={torch.__version__}")
    print(f"CUDA_AVAILABLE={torch.cuda.is_available()}")

    try:
        dml = torch_directml.device()
    except Exception as exc:  # pragma: no cover - direct environment probe
        print(f"DIRECTML_DEVICE_ERROR: {exc}")
        return 1

    print(f"DIRECTML_DEVICE={dml}")

    try:
        tensor1 = torch.tensor([1.0]).to(dml)
        tensor2 = torch.tensor([2.0]).to(dml)
        tensor_sum = tensor1 + tensor2
        print(f"TENSOR_SUM={tensor_sum.item()}")

        model = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 2),
        ).to(dml)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        inputs = torch.randn(16, 4).to(dml)
        targets = torch.randn(16, 2).to(dml)

        optimizer.zero_grad()
        predictions = model(inputs)
        loss = torch.nn.functional.mse_loss(predictions, targets)
        loss.backward()
        optimizer.step()
        print(f"TRAIN_STEP_LOSS={loss.item():.6f}")
    except Exception as exc:  # pragma: no cover - direct environment probe
        print(f"TRAIN_STEP_ERROR: {type(exc).__name__}: {exc}")
        return 1

    print("DIRECTML_SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
