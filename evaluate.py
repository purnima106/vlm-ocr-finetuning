# evaluate.py - Quantitative evaluation for OCR models

import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
import jiwer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"


def compute_bleu(reference, hypothesis):
    smooth = SmoothingFunction().method1
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smooth)


def compute_cer(reference, hypothesis):
    return jiwer.cer(reference, hypothesis)


def compute_wer(reference, hypothesis):
    return jiwer.wer(reference, hypothesis)


def load_base_model():
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    return model, processor


def load_finetuned_model(lora_path):
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    base_model = AutoModelForVision2Seq.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model = PeftModel.from_pretrained(base_model, lora_path)

    return model, processor


def run_inference(model, processor, image_path, prompt):

    image = Image.open(image_path).convert("RGB")

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

    inputs = processor(
        text=text,
        images=image,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256)

    prediction = processor.decode(output[0], skip_special_tokens=True)

    return prediction


def evaluate(model, processor, test_data):

    prompt = "Extract all text from this receipt."

    bleu_scores = []
    cer_scores = []
    wer_scores = []

    for sample in test_data:

        image_path = sample["image"]
        reference = sample["text"]

        prediction = run_inference(model, processor, image_path, prompt)

        bleu = compute_bleu(reference, prediction)
        cer = compute_cer(reference, prediction)
        wer = compute_wer(reference, prediction)

        bleu_scores.append(bleu)
        cer_scores.append(cer)
        wer_scores.append(wer)

    results = {
        "BLEU": sum(bleu_scores) / len(bleu_scores),
        "CER": sum(cer_scores) / len(cer_scores),
        "WER": sum(wer_scores) / len(wer_scores),
    }

    return results


def main():

    test_file = "dataset/test.json"
    lora_path = "outputs/qwen_ocr_lora"

    with open(test_file) as f:
        test_data = json.load(f)

    print("Loading base model...")
    base_model, base_processor = load_base_model()

    print("Evaluating base model...")
    base_results = evaluate(base_model, base_processor, test_data)

    print("\nBase Model Results")
    print(base_results)

    print("\nLoading fine-tuned model...")
    ft_model, ft_processor = load_finetuned_model(lora_path)

    print("Evaluating fine-tuned model...")
    ft_results = evaluate(ft_model, ft_processor, test_data)

    print("\nFine-tuned Model Results")
    print(ft_results)

    results = {
        "base_model": base_results,
        "finetuned_model": ft_results
    }

    Path("results").mkdir(exist_ok=True)

    with open("results/evaluation.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved results to results/evaluation.json")


if __name__ == "__main__":
    main()