import hashlib
import json
import statistics
from pathlib import Path
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = PROJECT_ROOT / "data" / "final" / "train.jsonl"
VALIDATION_FILE = PROJECT_ROOT / "data" / "final" / "validation.jsonl"

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_FILE = REPORT_DIR / "dataset_report.json"


def sha256_file(path: Path) -> str:
    sha256 = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(block)

    return sha256.hexdigest()


def load_jsonl(path: Path):
    examples = []
    errors = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append(
                    {
                        "line": line_number,
                        "error": str(exc),
                    }
                )

    return examples, errors


def validate_examples(examples):
    stats = {
        "missing_messages": 0,
        "messages_not_list": 0,
        "invalid_message_count": 0,
        "invalid_roles": 0,
        "empty_content": 0,
        "valid_examples": 0,
    }

    role_counter = Counter()

    questions = []
    answers = []
    canonical_examples = []

    for example in examples:

        messages = example.get("messages")

        if messages is None:
            stats["missing_messages"] += 1
            continue

        if not isinstance(messages, list):
            stats["messages_not_list"] += 1
            continue

        if len(messages) != 2:
            stats["invalid_message_count"] += 1
            continue

        valid = True

        for message in messages:

            if not isinstance(message, dict):
                valid = False
                stats["invalid_message_count"] += 1
                continue

            role = message.get("role")
            content = message.get("content")

            role_counter[role] += 1

            if role not in {"user", "assistant"}:
                stats["invalid_roles"] += 1
                valid = False

            if not isinstance(content, str) or not content.strip():
                stats["empty_content"] += 1
                valid = False

        if not valid:
            continue

        user_messages = [
            m["content"].strip()
            for m in messages
            if m["role"] == "user"
        ]

        assistant_messages = [
            m["content"].strip()
            for m in messages
            if m["role"] == "assistant"
        ]

        if len(user_messages) != 1 or len(assistant_messages) != 1:
            stats["invalid_message_count"] += 1
            continue

        question = user_messages[0]
        answer = assistant_messages[0]

        questions.append(question)
        answers.append(answer)

        canonical = json.dumps(
            {
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            },
            ensure_ascii=False,
            sort_keys=True,
        )

        canonical_examples.append(canonical)

        stats["valid_examples"] += 1

    return {
        "stats": stats,
        "roles": dict(role_counter),
        "questions": questions,
        "answers": answers,
        "canonical_examples": canonical_examples,
    }


def duplicate_count(items):
    counter = Counter(items)
    return sum(count - 1 for count in counter.values() if count > 1)


def duplicate_groups(items):
    counter = Counter(items)
    return sum(1 for count in counter.values() if count > 1)


def text_statistics(values):
    if not values:
        return {
            "count": 0,
            "min_chars": 0,
            "max_chars": 0,
            "mean_chars": 0,
            "median_chars": 0,
        }

    lengths = [len(value) for value in values]

    return {
        "count": len(values),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
        "mean_chars": round(statistics.mean(lengths), 2),
        "median_chars": round(statistics.median(lengths), 2),
    }


def analyze_dataset(name: str, path: Path):

    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")

    examples, json_errors = load_jsonl(path)
    validation = validate_examples(examples)

    return {
        "name": name,
        "path": str(path.relative_to(PROJECT_ROOT)),
        "file_size_bytes": path.stat().st_size,
        "file_size_mb": round(path.stat().st_size / (1024 * 1024), 4),
        "sha256": sha256_file(path),
        "total_json_lines": len(examples),
        "json_errors": json_errors,
        "validation": validation["stats"],
        "roles": validation["roles"],
        "questions": {
            "statistics": text_statistics(validation["questions"]),
            "exact_duplicate_groups": duplicate_groups(
                validation["questions"]
            ),
            "exact_duplicate_examples": duplicate_count(
                validation["questions"]
            ),
        },
        "answers": {
            "statistics": text_statistics(validation["answers"]),
            "exact_duplicate_groups": duplicate_groups(
                validation["answers"]
            ),
            "exact_duplicate_examples": duplicate_count(
                validation["answers"]
            ),
        },
        "_canonical_examples": validation["canonical_examples"],
        "_questions": validation["questions"],
    }


def main():

    print("=" * 72)
    print("UPTA-LLM DATASET VALIDATOR")
    print("=" * 72)

    train = analyze_dataset("train", TRAIN_FILE)
    validation = analyze_dataset("validation", VALIDATION_FILE)

    train_set = set(train["_canonical_examples"])
    validation_set = set(validation["_canonical_examples"])

    overlap = train_set.intersection(validation_set)

    train_questions = set(train["_questions"])
    validation_questions = set(validation["_questions"])

    question_overlap = train_questions.intersection(validation_questions)

    report = {
        "dataset": {
            "name": "UPTA-LLM Dataset",
            "version": "1.0",
            "format": "JSONL conversational messages",
        },
        "train": {
            key: value
            for key, value in train.items()
            if not key.startswith("_")
        },
        "validation": {
            key: value
            for key, value in validation.items()
            if not key.startswith("_")
        },
        "leakage": {
            "exact_example_overlap": len(overlap),
            "exact_question_overlap": len(question_overlap),
        },
        "totals": {
            "train_examples": train["validation"]["valid_examples"],
            "validation_examples": validation["validation"]["valid_examples"],
            "total_examples": (
                train["validation"]["valid_examples"]
                + validation["validation"]["valid_examples"]
            ),
        },
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with REPORT_FILE.open("w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("TRAIN")
    print("-" * 72)
    print(f"Ejemplos: {train['validation']['valid_examples']:,}")
    print(f"Tamaño:   {train['file_size_mb']:.2f} MB")
    print(f"SHA256:   {train['sha256']}")

    print()
    print("VALIDATION")
    print("-" * 72)
    print(f"Ejemplos: {validation['validation']['valid_examples']:,}")
    print(f"Tamaño:   {validation['file_size_mb']:.2f} MB")
    print(f"SHA256:   {validation['sha256']}")

    print()
    print("LEAKAGE")
    print("-" * 72)
    print(
        "Ejemplos idénticos train/validation:",
        len(overlap),
    )
    print(
        "Preguntas idénticas train/validation:",
        len(question_overlap),
    )

    print()
    print("PREGUNTAS")
    print("-" * 72)

    for name, dataset in [
        ("train", train),
        ("validation", validation),
    ]:
        stats = dataset["questions"]["statistics"]

        print(
            f"{name}: "
            f"min={stats['min_chars']} | "
            f"max={stats['max_chars']} | "
            f"mean={stats['mean_chars']} | "
            f"median={stats['median_chars']}"
        )

    print()
    print("RESPUESTAS")
    print("-" * 72)

    for name, dataset in [
        ("train", train),
        ("validation", validation),
    ]:
        stats = dataset["answers"]["statistics"]

        print(
            f"{name}: "
            f"min={stats['min_chars']} | "
            f"max={stats['max_chars']} | "
            f"mean={stats['mean_chars']} | "
            f"median={stats['median_chars']}"
        )

    print()
    print("REPORTE")
    print("-" * 72)
    print(REPORT_FILE)

    print()
    print("Validación terminada.")


if __name__ == "__main__":
    main()