"""
Compare base model and fine-tuned LoRA model for document OCR.

This script:
- loads the base Qwen2.5-VL-3B-Instruct model
- loads the LoRA adapter from adapter/ directory
- randomly selects 5-10 samples from data/test.json
- runs inference with both models
- prints a comparison showing ground truth vs base model vs fine-tuned output
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig


MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_PROMPT = "Extract all text from this document. Return only the text."


def load_base_model(
    model_name: str,
    cache_dir: str | None = None,
    use_4bit: bool = True,
) -> tuple[AutoModelForVision2Seq, AutoProcessor]:
    """
    Load the base Qwen2.5-VL model and processor.
    
    Args:
        model_name: HuggingFace model identifier
        cache_dir: Optional cache directory
        use_4bit: Whether to use 4-bit quantization
        
    Returns:
        Tuple of (model, processor)
    """
    print(f"Loading base model: {model_name}")
    
    processor = AutoProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    
    if use_4bit:
        # Use 4-bit quantization for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            device_map="auto",
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
    
    print("Base model loaded successfully")
    return model, processor


def load_finetuned_model(
    base_model: AutoModelForVision2Seq,
    adapter_path: str | Path,
) -> AutoModelForVision2Seq:
    """
    Load LoRA adapter weights onto the base model.
    
    Args:
        base_model: The base model
        adapter_path: Path to the adapter directory
        
    Returns:
        Model with LoRA adapter loaded
    """
    adapter_path = Path(adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")
    
    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    print("Fine-tuned model loaded successfully")
    return model


def load_test_samples(
    json_path: str | Path,
    num_samples: int = 5,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Load and randomly sample test examples.
    
    Args:
        json_path: Path to test.json
        num_samples: Number of samples to select
        seed: Random seed for reproducibility
        
    Returns:
        List of sampled test examples
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Test file not found: {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        all_samples = json.load(f)
    
    if len(all_samples) < num_samples:
        print(f"Warning: Only {len(all_samples)} samples available, using all of them")
        num_samples = len(all_samples)
    
    random.seed(seed)
    selected = random.sample(all_samples, num_samples)
    
    print(f"Selected {len(selected)} test samples")
    return selected


def extract_ground_truth(messages: List[Dict[str, Any]]) -> str:
    """
    Extract ground truth OCR text from messages.
    
    Args:
        messages: List of message dicts
        
    Returns:
        Ground truth text
    """
    for msg in messages:
        if msg["role"] == "assistant":
            for item in msg.get("content", []):
                if isinstance(item, dict) and item.get("type") == "text":
                    return item.get("text", "")
    return ""


def run_inference(
    model: AutoModelForVision2Seq,
    processor: AutoProcessor,
    image_path: str,
    prompt: str = DEFAULT_PROMPT,
) -> str:
    """
    Run inference on a single image.
    
    Args:
        model: The model to use for inference
        processor: The processor for the model
        image_path: Path to the image file
        prompt: Text prompt for the model
        
    Returns:
        Generated text output
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Format messages for Qwen2.5-VL
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    # Process inputs
    inputs = processor(
        text=[messages],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    
    # Move inputs to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )
    
    # Decode output
    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]
    
    return generated_text


def print_comparison(
    sample_id: str,
    ground_truth: str,
    base_output: str,
    finetuned_output: str,
) -> None:
    """
    Print a formatted comparison of outputs.
    
    Args:
        sample_id: Sample identifier
        ground_truth: Ground truth text
        base_output: Base model output
        finetuned_output: Fine-tuned model output
    """
    print("\n" + "=" * 80)
    print(f"Sample ID: {sample_id}")
    print("=" * 80)
    
    print("\n📄 GROUND TRUTH:")
    print("-" * 80)
    print(ground_truth)
    
    print("\n🔵 BASE MODEL OUTPUT:")
    print("-" * 80)
    print(base_output)
    
    print("\n🟢 FINE-TUNED MODEL OUTPUT:")
    print("-" * 80)
    print(finetuned_output)
    
    print("\n" + "=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare base and fine-tuned models for OCR"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="data/test.json",
        help="Path to test.json file",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default="adapter",
        help="Directory containing LoRA adapter weights",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of test samples to evaluate (5-10 recommended)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Prompt text for inference",
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Use 4-bit quantization for base model (saves memory)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for model files",
    )
    args = parser.parse_args()
    
    # Validate num_samples
    if args.num_samples < 1 or args.num_samples > 10:
        print(f"Warning: num_samples should be between 1 and 10, got {args.num_samples}")
        args.num_samples = max(1, min(10, args.num_samples))
    
    # Load test samples
    test_samples = load_test_samples(args.test_file, args.num_samples, args.seed)
    
    # Load base model
    base_model, processor = load_base_model(
        MODEL_NAME,
        cache_dir=args.cache_dir,
        use_4bit=args.use_4bit,
    )
    
    # Load fine-tuned model
    finetuned_model = load_finetuned_model(base_model, args.adapter_dir)
    
    # Set models to evaluation mode
    base_model.eval()
    finetuned_model.eval()
    
    print(f"\nRunning inference on {len(test_samples)} samples...")
    print("=" * 80)
    
    # Run inference on each sample
    for i, sample in enumerate(test_samples, 1):
        sample_id = sample.get("id", f"sample-{i}")
        image_path = sample.get("image", "")
        ground_truth = extract_ground_truth(sample.get("messages", []))
        
        if not Path(image_path).exists():
            print(f"\n⚠️  Warning: Image not found for {sample_id}: {image_path}")
            continue
        
        print(f"\n[{i}/{len(test_samples)}] Processing {sample_id}...")
        
        # Run inference with base model
        try:
            base_output = run_inference(base_model, processor, image_path, args.prompt)
        except Exception as e:
            print(f"❌ Error with base model: {e}")
            base_output = f"Error: {str(e)}"
        
        # Run inference with fine-tuned model
        try:
            finetuned_output = run_inference(
                finetuned_model, processor, image_path, args.prompt
            )
        except Exception as e:
            print(f"❌ Error with fine-tuned model: {e}")
            finetuned_output = f"Error: {str(e)}"
        
        # Print comparison
        print_comparison(sample_id, ground_truth, base_output, finetuned_output)
    
    print("\n✅ Comparison complete!")


if __name__ == "__main__":
    main()

