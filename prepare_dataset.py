"""
Prepare a small chat-style OCR dataset from HuggingFace CORD-v2.

This script:
- downloads the dataset with HuggingFace `datasets`
- extracts image + OCR ground-truth text
- converts samples into a simple chat format (user: image+prompt, assistant: text)
- randomly samples ~1000 train and ~50 test examples (or fewer if split is smaller)
- saves JSON files under ./data/ and images under ./data/images/

Note:
The dataset requested as "clovaai/cord-v2" may be moved/renamed on the Hub.
As of now, the accessible dataset is "naver-clova-ix/cord-v2".
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from datasets import concatenate_datasets, load_dataset


DATASET_NAME = "naver-clova-ix/cord-v2"
DEFAULT_PROMPT = "Extract all text from this document. Return only the text."


def extract_ocr_text(ground_truth: str) -> str:
    """
    Convert CORD-v2 `ground_truth` JSON string into plain OCR text.

    CORD-v2 stores rich annotations. For a straightforward OCR target, we use
    `valid_line` and join the `words[].text` tokens into lines.
    """
    try:
        gt = json.loads(ground_truth)
    except Exception:
        # If parsing fails for any reason, fall back to raw string.
        return str(ground_truth).strip()

    lines: List[str] = []
    for line in gt.get("valid_line", []) or []:
        words = line.get("words", []) or []
        tokens = [w.get("text", "") for w in words if str(w.get("text", "")).strip()]
        if tokens:
            lines.append(" ".join(tokens).strip())
    return "\n".join(lines).strip()


def sample_indices(n: int, k: int, seed: int) -> List[int]:
    k = min(k, n)
    rng = random.Random(seed)
    idxs = list(range(n))
    rng.shuffle(idxs)
    return idxs[:k]


def to_chat_record(sample_id: str, image_path: str, prompt: str, ocr_text: str) -> Dict[str, Any]:
    return {
        "id": sample_id,
        "image": image_path,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": ocr_text}]},
        ],
    }


def process_split(
    split_ds,
    out_images_dir: Path,
    prompt: str,
    max_samples: int,
    seed: int,
    split_name_for_id: str,
) -> List[Dict[str, Any]]:
    out_images_dir.mkdir(parents=True, exist_ok=True)

    idxs = sample_indices(len(split_ds), max_samples, seed)
    records: List[Dict[str, Any]] = []

    for i, idx in enumerate(idxs):
        ex = split_ds[int(idx)]

        # Save image to a stable local path so JSON can refer to it.
        # `datasets` returns PIL Images for Image features.
        img = ex["image"]
        img_filename = f"{split_name_for_id}_{i:06d}.png"
        img_path = out_images_dir / img_filename
        img.save(img_path)

        # Extract OCR target text from `ground_truth`.
        ocr_text = extract_ocr_text(ex.get("ground_truth", ""))

        rel_img_path = img_path.as_posix()
        sample_id = f"{split_name_for_id}-{i:06d}"
        records.append(to_chat_record(sample_id, rel_img_path, prompt, ocr_text))

    return records

#Entry point 
def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CORD-v2 chat-style OCR JSON files.")
    parser.add_argument("--train_size", type=int, default=1000, help="Number of training samples to write.")
    parser.add_argument("--test_size", type=int, default=50, help="Number of test samples to write.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="User prompt text for each example.")
    args = parser.parse_args()

    repo_dir = Path(__file__).resolve().parent
    data_dir = repo_dir / "data"
    images_dir = data_dir / "images"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download dataset (cached by HF). If downloads are slow, consider installing:
    #   pip install huggingface_hub[hf_xet]
    ds = load_dataset(DATASET_NAME)

    # CORD-v2 has train/validation/test splits (800/100/100).
    # To get closer to 1000 train samples, we pool train+validation.
    train_pool = concatenate_datasets([ds["train"], ds["validation"]])
    test_pool = ds["test"]

    train_records = process_split(
        train_pool,
        out_images_dir=images_dir / "train",
        prompt=args.prompt,
        max_samples=args.train_size,
        seed=args.seed,
        split_name_for_id="train",
    )
    test_records = process_split(
        test_pool,
        out_images_dir=images_dir / "test",
        prompt=args.prompt,
        max_samples=args.test_size,
        seed=args.seed + 1,  # keep split sampling independent but deterministic
        split_name_for_id="test",
    )

    train_json_path = data_dir / "train.json"
    test_json_path = data_dir / "test.json"

    train_json_path.write_text(json.dumps(train_records, ensure_ascii=False, indent=2), encoding="utf-8")
    test_json_path.write_text(json.dumps(test_records, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(train_records)} training records to: {train_json_path}")
    print(f"Wrote {len(test_records)} test records to: {test_json_path}")
    print(f"Images saved under: {images_dir}")


if __name__ == "__main__":
    main()


