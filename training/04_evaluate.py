import json
import os
import re
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel


BASE = "Qwen/Qwen3-8B"
ADAPTER = "outputs/qwen3-8b-upta"
VALIDATION = "data/final/validation.jsonl"
OUTPUT = "outputs/evaluation_validation.jsonl"


def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


print("=" * 70)
print("UPTA-LLM — EVALUACIÓN VALIDATION")
print("=" * 70)

print("\nCargando validation...")

ds = load_dataset(
    "json",
    data_files=VALIDATION,
)["train"]

print("Ejemplos:", len(ds))

print("\nConfigurando 4-bit...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("Cargando tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE,
    use_fast=True,
    trust_remote_code=True,
)

print("Cargando modelo base...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE,
    device_map="auto",
    dtype=torch.bfloat16,
    quantization_config=bnb_config,
    trust_remote_code=True,
)

print("Cargando adapter LoRA...")

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER,
)

model.eval()

print("\nModelo listo.")
print(
    "VRAM:",
    round(torch.cuda.memory_allocated() / 1024**3, 2),
    "GB"
)

os.makedirs("outputs", exist_ok=True)

resultados = []

for i, item in enumerate(ds):

    messages = item["messages"]

    pregunta = ""
    esperada = ""

    for msg in messages:
        if msg["role"] == "user":
            pregunta = msg["content"]
        elif msg["role"] == "assistant":
            esperada = msg["content"]

    prompt_messages = [
        {
            "role": "user",
            "content": pregunta,
        }
    ]

    prompt = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    # Qwen3 puede incluir bloques de razonamiento.
    generated_clean = re.sub(
        r"<think>.*?</think>",
        "",
        generated,
        flags=re.DOTALL,
    ).strip()

    resultados.append(
        {
            "index": i,
            "question": pregunta,
            "expected": esperada,
            "generated": generated_clean,
        }
    )

    if (i + 1) % 10 == 0:
        print(
            f"Evaluados: {i + 1}/{len(ds)}"
        )

with open(
    OUTPUT,
    "w",
    encoding="utf-8",
) as f:

    for item in resultados:
        f.write(
            json.dumps(
                item,
                ensure_ascii=False,
            )
            + "\n"
        )

print("\n" + "=" * 70)
print("EVALUACIÓN TERMINADA")
print("=" * 70)

print("Resultados:", OUTPUT)
print("Ejemplos:", len(resultados))

print("\nPRIMEROS 5 RESULTADOS")
print("=" * 70)

for item in resultados[:5]:

    print("\nPREGUNTA:")
    print(item["question"])

    print("\nESPERADA:")
    print(item["expected"])

    print("\nGENERADA:")
    print(item["generated"])

    print("-" * 70)
