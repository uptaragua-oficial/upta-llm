import os
import yaml
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE_DIR, "configs/model.yaml"), encoding="utf-8") as f:
    model_cfg = yaml.safe_load(f)

with open(os.path.join(BASE_DIR, "configs/train_qlora.yaml"), encoding="utf-8") as f:
    train_cfg = yaml.safe_load(f)

with open(os.path.join(BASE_DIR, "configs/dataset.yaml"), encoding="utf-8") as f:
    dataset_cfg = yaml.safe_load(f)


MODEL_NAME = model_cfg["model"]["name"]

TRAIN_FILE = os.path.join(
    BASE_DIR,
    dataset_cfg["dataset"]["train_file"]
)

VAL_FILE = os.path.join(
    BASE_DIR,
    dataset_cfg["dataset"]["validation_file"]
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    train_cfg["training"]["output_dir"]
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("=" * 70)
print("UPTA-LLM — QLoRA TRAINING")
print("=" * 70)

print("Model:", MODEL_NAME)
print("Train:", TRAIN_FILE)
print("Validation:", VAL_FILE)
print("Output:", OUTPUT_DIR)

print("\nGPU:", torch.cuda.get_device_name(0))
print(
    "VRAM:",
    round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
    "GB"
)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA no está disponible.")

if not torch.cuda.is_bf16_supported():
    raise RuntimeError("La GPU no soporta BF16.")

# ---------------------------------------------------------
# DATASET
# ---------------------------------------------------------

dataset = load_dataset(
    "json",
    data_files={
        "train": TRAIN_FILE,
        "validation": VAL_FILE,
    }
)

print("\nDataset:")
print(dataset)

# ---------------------------------------------------------
# TOKENIZER
# ---------------------------------------------------------

print("\nCargando tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---------------------------------------------------------
# QUANTIZATION
# ---------------------------------------------------------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

print("\nCargando modelo...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

model.config.use_cache = False

print(
    "VRAM después de cargar modelo:",
    round(torch.cuda.memory_allocated() / 1024**3, 2),
    "GB"
)

# ---------------------------------------------------------
# LORA
# ---------------------------------------------------------

lora_cfg = model_cfg["model"]["lora"]

peft_config = LoraConfig(
    r=lora_cfg["rank"],
    lora_alpha=lora_cfg["alpha"],
    lora_dropout=lora_cfg["dropout"],
    bias=lora_cfg["bias"],
    task_type="CAUSAL_LM",
    target_modules=lora_cfg["target_modules"],
)

# ---------------------------------------------------------
# TRAINING CONFIG
# ---------------------------------------------------------

t = train_cfg["training"]

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,

    num_train_epochs=t["num_train_epochs"],
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    gradient_accumulation_steps=4,

    learning_rate=t["learning_rate"],
    weight_decay=t["weight_decay"],

    max_length=t["max_seq_length"],

    logging_steps=t["logging_steps"],

    save_strategy="steps",
    save_steps=t["save_steps"],
    save_total_limit=t["save_total_limit"],

    eval_strategy="steps",
    eval_steps=t["eval_steps"],

    gradient_checkpointing=True,

    bf16=True,
    fp16=False,

    optim=t["optim"],

    warmup_steps=10,
    lr_scheduler_type=t["lr_scheduler_type"],

    report_to="none",

    packing=False,

    dataset_text_field=None,

    seed=t["seed"],
)

# ---------------------------------------------------------
# TRAINER
# ---------------------------------------------------------

trainer = SFTTrainer(
    model=model,
    args=training_args,

    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],

    processing_class=tokenizer,

    peft_config=peft_config,
)

print("\nTrainer creado correctamente.")

print(
    "\nParámetros entrenables:",
    sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
)

# ---------------------------------------------------------
# TRAIN
# ---------------------------------------------------------

print("\nIniciando entrenamiento...")

trainer.train()

print("\nGuardando adapter LoRA...")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n" + "=" * 70)
print("ENTRENAMIENTO FINALIZADO")
print("=" * 70)
print("Salida:", OUTPUT_DIR)
