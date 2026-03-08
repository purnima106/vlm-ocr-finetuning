# Vision-Language Model Fine-tuning for OCR

A research project demonstrating fine-tuning of Qwen2.5-VL-3B-Instruct for document OCR using QLoRA (Quantized Low-Rank Adaptation). This project uses the CORD-v2 receipt dataset to train a vision-language model that can extract text from document images.

## Project Overview

This project implements an end-to-end pipeline for fine-tuning a vision-language model on OCR tasks:

1. **Dataset Preparation**: Downloads and processes the CORD-v2 dataset into a chat-style format
2. **Model Fine-tuning**: Applies QLoRA to efficiently fine-tune Qwen2.5-VL-3B-Instruct
3. **Evaluation**: Compares base model vs fine-tuned model performance on test samples

The approach uses 4-bit quantization and LoRA adapters to enable efficient training on consumer hardware while maintaining model performance.

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for training)
- ~10GB disk space for model and dataset

### Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note**: `bitsandbytes` requires CUDA. If you encounter installation issues, refer to the [bitsandbytes documentation](https://github.com/TimDettmers/bitsandbytes).

## Dataset Description

The project uses the **CORD-v2** dataset (Consolidated Receipt Dataset), which contains:
- Receipt images with OCR ground truth annotations
- Structured text extraction from receipts
- Train/validation/test splits (approximately 800/100/100 samples)

The dataset is automatically downloaded from HuggingFace during the preparation step. The preparation script extracts OCR text from the structured annotations and converts samples into a chat-style format suitable for vision-language model training.

## Training Configuration

The fine-tuning process uses the following configuration:

- **Model**: Qwen2.5-VL-3B-Instruct
- **Quantization**: 4-bit (NF4) via bitsandbytes
- **LoRA**: Rank 16, Alpha 32, Dropout 0.05
- **Training**: ~300 steps with batch size 1, gradient accumulation steps 4
- **Optimizer**: paged_adamw_8bit (memory-efficient)
- **Learning Rate**: 2e-4 with 10 warmup steps

Only LoRA adapter weights are saved (~50MB), not the full model, making the approach memory-efficient.

## Usage

### 1. Prepare Dataset

Download and process the CORD-v2 dataset:

```bash
python prepare_dataset.py
```

This will:
- Download the dataset from HuggingFace
- Extract OCR text from annotations
- Sample ~1000 training and ~50 test examples
- Save processed data to `data/train.json` and `data/test.json`
- Save images to `data/images/`

**Options**:
- `--train_size`: Number of training samples (default: 1000)
- `--test_size`: Number of test samples (default: 50)
- `--seed`: Random seed for sampling (default: 42)

### 2. Fine-tune Model

Train the model with QLoRA:

```bash
python finetune.py
```

This will:
- Load the base model with 4-bit quantization
- Apply LoRA adapters
- Train on the prepared dataset
- Save adapter weights to `adapter/`

**Options**:
- `--dataset`: Path to training JSON (default: `data/train.json`)
- `--output_dir`: Directory to save adapters (default: `adapter`)
- `--max_steps`: Training steps (default: 300)
- `--batch_size`: Batch size (default: 1)
- `--gradient_accumulation_steps`: Gradient accumulation (default: 4)
- `--learning_rate`: Learning rate (default: 2e-4)
- `--logging_steps`: Logging frequency (default: 10)

### 3. Run Inference Comparison

Compare base model vs fine-tuned model:

```bash
python inference.py
```

This will:
- Load both base and fine-tuned models
- Randomly select 5-10 test samples
- Run inference with both models
- Display side-by-side comparison

**Options**:
- `--test_file`: Path to test JSON (default: `data/test.json`)
- `--adapter_dir`: Adapter directory (default: `adapter`)
- `--num_samples`: Number of samples to evaluate (default: 5)
- `--seed`: Random seed for sample selection (default: 42)
- `--use_4bit`: Use 4-bit quantization for base model (saves memory)

## Results Discussion

The fine-tuning process adapts the pre-trained Qwen2.5-VL model to better understand document OCR tasks. Key observations:

- **Efficiency**: QLoRA enables fine-tuning with minimal memory overhead (~50MB adapter weights vs ~6GB full model)
- **Performance**: The fine-tuned model should show improved OCR accuracy on receipt-style documents compared to the base model
- **Generalization**: The model learns to extract structured text from document images while maintaining the base model's general vision-language capabilities

The comparison script (`inference.py`) provides a direct way to evaluate improvements by showing ground truth, base model output, and fine-tuned model output side-by-side.

## Project Structure

```
vlm_ocr_finetuning/
├── prepare_dataset.py    # Dataset preparation script
├── finetune.py           # Training script with QLoRA
├── inference.py          # Model comparison script
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── data/                 # Processed dataset
│   ├── train.json
│   ├── test.json
│   └── images/
└── adapter/              # Saved LoRA adapter weights (after training)
```

## Notes

- The dataset identifier may change on HuggingFace. The script uses `naver-clova-ix/cord-v2` (the current accessible version).
- Training time depends on hardware. On a modern GPU, ~300 steps typically takes 1-3 hours.
- For best results, adjust training steps, learning rate, and LoRA parameters based on your specific use case.

## License

This project uses the Qwen2.5-VL model and CORD-v2 dataset. Please refer to their respective licenses:
- [Qwen2.5-VL License](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [CORD-v2 Dataset License](https://huggingface.co/datasets/naver-clova-ix/cord-v2)

