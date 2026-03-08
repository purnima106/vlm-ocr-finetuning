"""
Fine-tune Qwen2.5-VL-3B-Instruct for OCR using QLoRA.

This script:
- loads Qwen2.5-VL-3B-Instruct from HuggingFace
- applies 4-bit quantization with bitsandbytes
- adds LoRA adapters using PEFT
- loads the chat-style dataset from data/train.json
- trains for ~300 steps with batch size 1 and gradient accumulation
- saves only the LoRA adapter weights in adapter/
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


# Model configuration
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

# LoRA configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training configuration
DEFAULT_MAX_STEPS = 300
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 4
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_WARMUP_STEPS = 10


def load_model_and_processor(
    model_name: str,
    cache_dir: str | None = None,
) -> tuple[AutoModelForVision2Seq, AutoProcessor]:
    """
    Load the Qwen2.5-VL model and processor with 4-bit quantization.
    
    Args:
        model_name: HuggingFace model identifier
        cache_dir: Optional cache directory for model files
        
    Returns:
        Tuple of (model, processor)
    """
    print(f"Loading model and processor: {model_name}")
    
    # Configure 4-bit quantization with bitsandbytes
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load processor (handles images and text tokenization)
    processor = AutoProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    
    # Load model with 4-bit quantization
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    
    # Prepare model for k-bit training (enables gradient checkpointing, etc.)
    model = prepare_model_for_kbit_training(model)
    
    print("Model loaded successfully with 4-bit quantization")
    return model, processor


def setup_lora(model: AutoModelForVision2Seq) -> AutoModelForVision2Seq:
    """
    Configure and apply LoRA adapters to the model.
    
    Args:
        model: The quantized model
        
    Returns:
        Model with LoRA adapters applied
    """
    print("Setting up LoRA adapters...")
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",  # Causal language modeling
    )
    
    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    print("LoRA adapters configured successfully")
    return model


def load_dataset(json_path: str | Path) -> Dataset:
    """
    Load the training dataset from JSON file.
    
    Args:
        json_path: Path to train.json
        
    Returns:
        HuggingFace Dataset object
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {json_path}")
    
    print(f"Loading dataset from: {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} examples")
    return Dataset.from_list(data)


def format_messages_for_qwen(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format messages from our dataset format to Qwen2.5-VL format.
    
    Our dataset format has:
    - user: content = [{"type": "image", "image": path}, {"type": "text", "text": "..."}]
    - assistant: content = [{"type": "text", "text": "..."}]
    
    Qwen2.5-VL processor expects messages in this format:
    - [{"role": "user", "content": [{"type": "image", "image": PIL.Image}, {"type": "text", "text": "..."}]}, ...]
    - The processor will handle image loading if paths are provided, but we load images separately
    """
    formatted = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        
        # Convert image paths to placeholder - images are loaded separately in preprocess_function
        # The processor will replace these with actual image tensors
        processed_content = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "image":
                    # Keep image path - processor can handle it, or we'll replace it with loaded image
                    processed_content.append(item)
                elif item.get("type") == "text":
                    processed_content.append(item)
        
        formatted.append({
            "role": role,
            "content": processed_content if processed_content else content,
        })
    
    return formatted


def preprocess_function(examples: Dict[str, Any], processor: AutoProcessor) -> Dict[str, Any]:
    """
    Preprocess a batch of examples for training.
    
    Args:
        examples: Batch of examples from the dataset (from Dataset.map)
        processor: Qwen2.5-VL processor
        
    Returns:
        Processed batch with input_ids, attention_mask, labels, etc.
    """
    # Dataset.map passes batches, so examples is a dict with lists
    messages_list = examples["messages"]
    image_paths = examples.get("image", [""] * len(messages_list))
    
    texts = []
    images = []
    
    for messages, img_path in zip(messages_list, image_paths):
        # Load image - prefer the image field, fall back to extracting from messages
        image = None
        image_path_to_use = img_path
        
        if img_path and Path(img_path).exists():
            image = Image.open(img_path).convert("RGB")
        else:
            # Extract image path from user message content
            for msg in messages:
                if msg["role"] == "user":
                    for item in msg.get("content", []):
                        if isinstance(item, dict) and item.get("type") == "image":
                            img_path_from_msg = item.get("image", "")
                            if img_path_from_msg and Path(img_path_from_msg).exists():
                                image = Image.open(img_path_from_msg).convert("RGB")
                                image_path_to_use = img_path_from_msg
                                break
                    if image:
                        break
        
        if image is None:
            raise ValueError(f"Could not load image. Path: {img_path}")
        
        images.append(image)
        
        # Format messages for Qwen2.5-VL processor
        # Replace image paths in messages with actual PIL Images
        formatted_messages = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            processed_content = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "image":
                        # Replace image path with actual PIL Image for the processor
                        processed_content.append({"type": "image", "image": image})
                    elif item.get("type") == "text":
                        processed_content.append(item)
            
            formatted_messages.append({
                "role": role,
                "content": processed_content if processed_content else content,
            })
        
        texts.append(formatted_messages)
    
    # Process with the processor
    # Qwen2.5-VL processor handles tokenization and image preprocessing
    # Images are already embedded in the messages, so we only pass texts
    model_inputs = processor(
        text=texts,
        padding=True,
        return_tensors="pt",
    )
    
    # Create labels for training (same as input_ids but masked for user tokens)
    labels = model_inputs["input_ids"].clone()
    
    # Mask padding tokens in labels (set to -100 so they're ignored in loss)
    if processor.tokenizer.pad_token_id is not None:
        labels[labels == processor.tokenizer.pad_token_id] = -100
    
    model_inputs["labels"] = labels
    
    return model_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-VL-3B-Instruct with QLoRA for OCR")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/train.json",
        help="Path to training dataset JSON file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="adapter",
        help="Directory to save LoRA adapter weights",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Maximum number of training steps",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Training batch size",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=DEFAULT_WARMUP_STEPS,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=50,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log training metrics every N steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for model files",
    )
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load model and processor with 4-bit quantization
    model, processor = load_model_and_processor(MODEL_NAME, cache_dir=args.cache_dir)
    
    # Step 2: Setup LoRA adapters
    model = setup_lora(model)
    
    # Step 3: Load dataset
    dataset = load_dataset(args.dataset)
    
    # Step 4: Preprocess dataset
    print("Preprocessing dataset...")
    processed_dataset = dataset.map(
        lambda x: preprocess_function(x, processor),
        batched=True,
        batch_size=args.batch_size,
        remove_columns=dataset.column_names,
    )
    
    # Step 5: Configure training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        logging_strategy="steps",
        fp16=True,  # Use mixed precision training
        optim="paged_adamw_8bit",  # Memory-efficient optimizer
        remove_unused_columns=False,
        report_to="none",  # Disable wandb/tensorboard by default
        seed=args.seed,
    )
    
    # Step 6: Create data collator
    # For vision-language models, we use a simple collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=processor.tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Step 7: Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        data_collator=data_collator,
    )
    
    # Step 8: Train
    print("Starting training...")
    print(f"Training for {args.max_steps} steps")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    
    trainer.train()
    
    # Step 9: Save final adapter
    print(f"Saving LoRA adapter to: {output_dir}")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    
    print("Training completed successfully!")
    print(f"LoRA adapter saved in: {output_dir}")


if __name__ == "__main__":
    main()

