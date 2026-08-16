#!/usr/bin/env python3

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)

from trl import SFTConfig, SFTTrainer


# ============================================================
# Helpers
# ============================================================


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_project_path(path):
    """
    Resolve paths relative to the project root.
    """
    project_root = Path(__file__).resolve().parents[1]
    return project_root / path


def print_section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def check_cuda():
    print_section("CUDA / GPU")

    print("PyTorch:", torch.__version__)
    print("CUDA runtime:", torch.version.cuda)
    print("CUDA disponible:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA no está disponible. "
            "El entrenamiento requiere una GPU NVIDIA."
        )

    print("GPU:", torch.cuda.get_device_name(0))

    props = torch.cuda.get_device_properties(0)

    print(
        "VRAM:",
        f"{props.total_memory / (1024 ** 3):.2f} GB",
    )

    print(
        "Compute capability:",
        f"{props.major}.{props.minor}",
    )

    print(
        "BF16 soportado:",
        torch.cuda.is_bf16_supported(),
    )


def check_dataset(dataset):
    print_section("DATASET")

    print("Train:", len(dataset["train"]))

    if "validation" in dataset:
        print("Validation:", len(dataset["validation"]))

    example = dataset["train"][0]

    print()
    print("Primer registro:")

    print(
        json.dumps(
            example,
            ensure_ascii=False,
            indent=2,
        )
    )


# ============================================================
# Chat template
# ============================================================


def prepare_chat_dataset(dataset, tokenizer):
    """
    Converts the messages structure into the text representation
    expected by SFTTrainer.

    The original dataset remains unchanged.
    """

    def format_example(example):
        messages = example["messages"]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        return {
            "text": text
        }

    return dataset.map(
        format_example,
        desc="Aplicando chat template",
    )


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="UPTA-LLM QLoRA training"
    )

    parser.add_argument(
        "--model-config",
        default="configs/model.yaml",
    )

    parser.add_argument(
        "--dataset-config",
        default="configs/dataset.yaml",
    )

    parser.add_argument(
        "--train-config",
        default="configs/train_qlora.yaml",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Project paths
    # --------------------------------------------------------

    model_config_path = resolve_project_path(
        args.model_config
    )

    dataset_config_path = resolve_project_path(
        args.dataset_config
    )

    train_config_path = resolve_project_path(
        args.train_config
    )

    model_cfg = load_yaml(model_config_path)
    dataset_cfg = load_yaml(dataset_config_path)
    train_cfg = load_yaml(train_config_path)

    model_name = model_cfg["model"]["name"]

    dataset_settings = dataset_cfg["dataset"]
    training_settings = train_cfg["training"]
    lora_settings = train_cfg["lora"]
    quant_settings = train_cfg["quantization"]

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    seed = training_settings.get(
        "seed",
        42,
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print_section(
        "UPTA-LLM - QLoRA TRAINING"
    )

    print("Modelo:", model_name)
    print("Método:", training_settings["method"])
    print("Framework:", training_settings["framework"])
    print("Seed:", seed)

    # --------------------------------------------------------
    # CUDA
    # --------------------------------------------------------

    check_cuda()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_file = resolve_project_path(
        dataset_settings["train_file"]
    )

    validation_file = resolve_project_path(
        dataset_settings["validation_file"]
    )

    if not train_file.exists():
        raise FileNotFoundError(
            f"No existe el dataset de entrenamiento: {train_file}"
        )

    if not validation_file.exists():
        raise FileNotFoundError(
            f"No existe el dataset de validación: {validation_file}"
        )

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_file),
            "validation": str(validation_file),
        },
    )

    check_dataset(dataset)

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    print_section("TOKENIZER")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=model_cfg["model"]["tokenizer"]["use_fast"],
        trust_remote_code=model_cfg["model"]["trust_remote_code"],
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizer cargado.")
    print("Vocab size:", tokenizer.vocab_size)
    print("Pad token:", tokenizer.pad_token)
    print("EOS token:", tokenizer.eos_token)

    # --------------------------------------------------------
    # Chat template
    # --------------------------------------------------------

    print_section("CHAT TEMPLATE")

    dataset = prepare_chat_dataset(
        dataset,
        tokenizer,
    )

    print("Ejemplo formateado:")
    print()
    print(dataset["train"][0]["text"])

    # --------------------------------------------------------
    # Quantization
    # --------------------------------------------------------

    print_section("4-BIT QUANTIZATION")

    compute_dtype = torch.bfloat16

    if quant_settings["compute_dtype"] == "float16":
        compute_dtype = torch.float16

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=quant_settings["load_in_4bit"],
        bnb_4bit_quant_type=quant_settings["quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=quant_settings[
            "use_double_quant"
        ],
    )

    print("4-bit:", quant_settings["load_in_4bit"])
    print("Type:", quant_settings["quant_type"])
    print("Compute dtype:", compute_dtype)
    print(
        "Double quant:",
        quant_settings["use_double_quant"],
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print_section("MODEL")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=compute_dtype,
        trust_remote_code=model_cfg["model"]["trust_remote_code"],
    )

    print("Modelo cargado correctamente.")

    # --------------------------------------------------------
    # Prepare k-bit training
    # --------------------------------------------------------

    model = prepare_model_for_kbit_training(
        model
    )

    model.config.use_cache = False

    # --------------------------------------------------------
    # LoRA
    # --------------------------------------------------------

    print_section("LoRA")

    target_modules = model_cfg["model"]["lora"][
        "target_modules"
    ]

    lora_config = LoraConfig(
        r=lora_settings["r"],
        lora_alpha=lora_settings["alpha"],
        lora_dropout=lora_settings["dropout"],
        bias=model_cfg["model"]["lora"]["bias"],
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    print("Rank:", lora_settings["r"])
    print("Alpha:", lora_settings["alpha"])
    print("Dropout:", lora_settings["dropout"])

    print("Target modules:")

    for module in target_modules:
        print("  -", module)

    # --------------------------------------------------------
    # Training configuration
    # --------------------------------------------------------

    print_section("TRAINING CONFIGURATION")

    output_dir = resolve_project_path(
        training_settings["output_dir"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    max_seq_length = training_settings[
        "max_seq_length"
    ]

    # --------------------------------------------------------
    # TRL compatibility
    # --------------------------------------------------------

    sft_kwargs = dict(
        output_dir=str(output_dir),

        num_train_epochs=training_settings[
            "num_train_epochs"
        ],

        per_device_train_batch_size=training_settings[
            "per_device_train_batch_size"
        ],

        per_device_eval_batch_size=training_settings[
            "per_device_eval_batch_size"
        ],

        gradient_accumulation_steps=training_settings[
            "gradient_accumulation_steps"
        ],

        learning_rate=training_settings[
            "learning_rate"
        ],

        weight_decay=training_settings[
            "weight_decay"
        ],

        logging_steps=training_settings[
            "logging_steps"
        ],

        save_strategy=training_settings[
            "save_strategy"
        ],

        save_steps=training_settings[
            "save_steps"
        ],

        save_total_limit=training_settings[
            "save_total_limit"
        ],

        eval_strategy=training_settings[
            "evaluation_strategy"
        ],

        eval_steps=training_settings[
            "eval_steps"
        ],

        gradient_checkpointing=training_settings[
            "gradient_checkpointing"
        ],

        bf16=training_settings["bf16"],
        fp16=training_settings["fp16"],

        optim=training_settings["optim"],

        lr_scheduler_type=training_settings[
            "lr_scheduler_type"
        ],

        warmup_ratio=training_settings[
            "warmup_ratio"
        ],

        report_to=training_settings[
            "report_to"
        ],

        seed=seed,

        max_length=max_seq_length,

        packing=training_settings[
            "packing"
        ],

        dataset_text_field="text",

        dataset_num_proc=None,

        remove_unused_columns=False,
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    print_section("INITIALIZING SFT TRAINER")

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            **sft_kwargs,
        ),
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print("SFTTrainer creado correctamente.")

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print_section("STARTING TRAINING")

    trainer.train()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print_section("SAVING ADAPTER")

    trainer.save_model(
        str(output_dir)
    )

    tokenizer.save_pretrained(
        str(output_dir)
    )

    print()
    print("Modelo LoRA guardado en:")
    print(output_dir)

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("TRAINING COMPLETED")
    print("=" * 72)

    if torch.cuda.is_available():

        allocated = (
            torch.cuda.memory_allocated(0)
            / (1024 ** 3)
        )

        reserved = (
            torch.cuda.memory_reserved(0)
            / (1024 ** 3)
        )

        print(
            f"GPU memory allocated: {allocated:.2f} GB"
        )

        print(
            f"GPU memory reserved: {reserved:.2f} GB"
        )


if __name__ == "__main__":
    main()