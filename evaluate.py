"""
evaluate.py

Compute OCR metrics comparing base model and fine-tuned model.
"""

import json
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)

import jiwer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"


def compute_bleu(reference, prediction):

    smooth = SmoothingFunction().method1

    ref_tokens = reference.split()
    pred_tokens = prediction.split()

    return sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth)


def compute_cer(reference, prediction):

    return jiwer.cer(reference, prediction)


def compute_wer(reference, prediction):

    return jiwer.wer(reference, prediction)


def load_model(use_4bit=True):

    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    if use_4bit:

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
        )

    else:

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            device_map="auto",
        )

    return model, processor


def run_ocr(model, processor, image_path):

    image = Image.open(image_path).convert("RGB")

    prompt = "Extract all text from this document."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False)

    inputs = processor(text=text, images=image, return_tensors="pt")

    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512)

    prediction = processor.decode(outputs[0], skip_special_tokens=True)

    return prediction


def evaluate(test_file):

    with open(test_file) as f:
        data = json.load(f)

    model, processor = load_model()

    bleu_scores = []
    cer_scores = []
    wer_scores = []

    for sample in data:

        image = sample["image"]

        gt = ""
        for msg in sample["messages"]:
            if msg["role"] == "assistant":
                gt = msg["content"][0]["text"]

        pred = run_ocr(model, processor, image)

        bleu_scores.append(compute_bleu(gt, pred))
        cer_scores.append(compute_cer(gt, pred))
        wer_scores.append(compute_wer(gt, pred))

    print("\nEvaluation Results\n")

    print("BLEU:", sum(bleu_scores) / len(bleu_scores))
    print("CER :", sum(cer_scores) / len(cer_scores))
    print("WER :", sum(wer_scores) / len(wer_scores))


if __name__ == "__main__":

    evaluate("data/test.json")