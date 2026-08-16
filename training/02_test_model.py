#!/usr/bin/env python3

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_ID = "Qwen/Qwen3-8B"


def main():
    print("=" * 72)
    print("UPTA-LLM - QWEN3-8B SMOKE TEST")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 1. CUDA
    # ------------------------------------------------------------------

    print("\n[1] CUDA")
    print("-" * 72)

    print("PyTorch:", torch.__version__)
    print("CUDA runtime:", torch.version.cuda)
    print("CUDA disponible:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        print("\nERROR: CUDA no está disponible.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)

    print("GPU:", gpu_name)
    print(
        "VRAM:",
        f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB",
    )
    print(
        "Compute capability:",
        f"{torch.cuda.get_device_capability(0)[0]}."
        f"{torch.cuda.get_device_capability(0)[1]}",
    )
    print(
        "BF16 soportado:",
        torch.cuda.is_bf16_supported(),
    )

    # ------------------------------------------------------------------
    # 2. bitsandbytes
    # ------------------------------------------------------------------

    print("\n[2] bitsandbytes")
    print("-" * 72)

    try:
        import bitsandbytes as bnb

        print("bitsandbytes:", bnb.__version__)
    except Exception as exc:
        print("ERROR cargando bitsandbytes:", exc)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Quantization configuration
    # ------------------------------------------------------------------

    print("\n[3] Configuración 4-bit")
    print("-" * 72)

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    print("Compute dtype:", compute_dtype)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    print("load_in_4bit: True")
    print("quant_type: nf4")
    print("double_quant: True")

    # ------------------------------------------------------------------
    # 4. Tokenizer
    # ------------------------------------------------------------------

    print("\n[4] Tokenizer")
    print("-" * 72)

    print("Modelo:", MODEL_ID)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    print("Tokenizer cargado correctamente.")
    print("Vocab size:", tokenizer.vocab_size)

    # ------------------------------------------------------------------
    # 5. Model
    # ------------------------------------------------------------------

    print("\n[5] Cargando modelo")
    print("-" * 72)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )

    print("Modelo cargado correctamente.")

    # ------------------------------------------------------------------
    # 6. Model information
    # ------------------------------------------------------------------

    print("\n[6] Información del modelo")
    print("-" * 72)

    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        "Parámetros totales:",
        f"{total_params:,}",
    )

    print(
        "Parámetros entrenables:",
        f"{trainable_params:,}",
    )

    print(
        "Dtype del modelo:",
        next(model.parameters()).dtype,
    )

    print("Device map:")

    if hasattr(model, "hf_device_map"):
        print(model.hf_device_map)

    # ------------------------------------------------------------------
    # 7. Simple inference
    # ------------------------------------------------------------------

    print("\n[7] Inferencia")
    print("-" * 72)

    prompt = (
        "¿Qué es la Universidad Politécnica Territorial "
        "del Estado Aragua Federico Brito Figueroa?"
    )

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    print("Prompt:")
    print(prompt)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    print("\nRespuesta:")
    print(response)

    # ------------------------------------------------------------------
    # 8. GPU memory
    # ------------------------------------------------------------------

    print("\n[8] Memoria GPU")
    print("-" * 72)

    allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)

    print(f"Memoria asignada: {allocated:.2f} GB")
    print(f"Memoria reservada: {reserved:.2f} GB")

    print("\n" + "=" * 72)
    print("SMOKE TEST COMPLETADO CORRECTAMENTE")
    print("=" * 72)


if __name__ == "__main__":
    main()