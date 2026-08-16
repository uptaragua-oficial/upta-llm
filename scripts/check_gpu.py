#!/usr/bin/env python3

import subprocess
import sys


def main():
    print("=" * 72)
    print("UPTA-LLM - GPU CHECK")
    print("=" * 72)

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch no está instalado.")
        sys.exit(1)

    print()
    print("PyTorch:", torch.__version__)
    print("CUDA runtime:", torch.version.cuda)
    print("CUDA disponible:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        print()
        print("ERROR: CUDA no está disponible.")
        sys.exit(1)

    print()
    print("GPU")
    print("-" * 72)

    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)

        print(f"GPU {index}:")
        print("  Nombre:", props.name)
        print(
            "  VRAM:",
            f"{props.total_memory / (1024 ** 3):.2f} GB",
        )
        print(
            "  Compute capability:",
            f"{props.major}.{props.minor}",
        )

    print()
    print("CAPACIDADES")
    print("-" * 72)
    print("BF16:", torch.cuda.is_bf16_supported())

    print()
    print("NVIDIA-SMI")
    print("-" * 72)

    subprocess.run(
        ["nvidia-smi"],
        check=False,
    )

    print()
    print("=" * 72)


if __name__ == "__main__":
    main()