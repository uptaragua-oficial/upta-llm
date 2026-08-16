#!/usr/bin/env python3

import importlib
import platform
import shutil
import subprocess
import sys


PACKAGES = [
    ("torch", "PyTorch"),
    ("transformers", "Transformers"),
    ("peft", "PEFT"),
    ("trl", "TRL"),
    ("bitsandbytes", "bitsandbytes"),
    ("accelerate", "Accelerate"),
    ("datasets", "Datasets"),
    ("yaml", "PyYAML"),
]


def version(package_name):
    try:
        module = importlib.import_module(package_name)
        return getattr(module, "__version__", "versión no disponible")
    except Exception as exc:
        return f"NO DISPONIBLE ({exc})"


def command_exists(command):
    return shutil.which(command) is not None


def main():
    print("=" * 72)
    print("UPTA-LLM - ENVIRONMENT CHECK")
    print("=" * 72)

    print()
    print("SISTEMA")
    print("-" * 72)
    print("Python:", sys.version.replace("\n", " "))
    print("Executable:", sys.executable)
    print("Platform:", platform.platform())

    print()
    print("COMANDOS")
    print("-" * 72)
    print("git:", command_exists("git"))
    print("nvidia-smi:", command_exists("nvidia-smi"))

    print()
    print("PYTHON PACKAGES")
    print("-" * 72)

    for package, label in PACKAGES:
        print(f"{label:18}: {version(package)}")

    print()
    print("PYTORCH / CUDA")
    print("-" * 72)

    try:
        import torch

        print("PyTorch:", torch.__version__)
        print("CUDA runtime:", torch.version.cuda)
        print("CUDA disponible:", torch.cuda.is_available())

        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
            print("GPU count:", torch.cuda.device_count())

            props = torch.cuda.get_device_properties(0)

            print(
                "VRAM:",
                f"{props.total_memory / (1024 ** 3):.2f} GB",
            )

            print(
                "Compute capability:",
                f"{props.major}.{props.minor}",
            )

            print("BF16 soportado:", torch.cuda.is_bf16_supported())

    except Exception as exc:
        print("ERROR:", exc)

    print()
    print("NVIDIA-SMI")
    print("-" * 72)

    if command_exists("nvidia-smi"):
        try:
            subprocess.run(
                ["nvidia-smi"],
                check=False,
            )
        except Exception as exc:
            print("No se pudo ejecutar nvidia-smi:", exc)
    else:
        print("nvidia-smi no está disponible.")

    print()
    print("=" * 72)


if __name__ == "__main__":
    main()