# SafetyBERT QA: Active Learning for Workplace Safety
This repository provides script for training and implementation for SafetyBERT QA using Active Learning Approach in Safety domain

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow)](https://huggingface.co/adanish91/safety-qa-bert)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97-Datasets-yellow)](https://huggingface.co/datasets/adanish91/safety-qa-bert-dataset)

Training code and implementation for **SafetyBERT QA**, a question-answering system for workplace safety incidents using active learning and BERT fine-tuning.

## Overview

This repository contains the implementation of an active learning-based question answering system specifically designed for occupational safety and health domains. The model is fine-tuned from SafetyBERT to answer questions about workplace safety incidents.

## Key Features

- **Active Learning**: Efficient annotation through uncertainty-based sample selection
- **SafetyBERT Base**: Fine-tuned from domain-specific SafetyBERT model
- **Real-World Data**: Trained on MSHA (Mine Safety and Health Administration) incident narratives
- **SQuAD Format**: Converts safety narratives to extractive QA format
- **Multi-Question Types**: Handles body part, work activity, and accident cause questions

## Quick Start

### Installation

Clone the repository
git clone https://github.com/yourusername/safety-qa-bert.git
cd safety-qa-bert

### Using Pre-trained Model

from transformers import AutoTokenizer, AutoModelForQuestionAnswering

Load model from Hugging Face
model = AutoModelForQuestionAnswering.from_pretrained("adanish91/safety-qa-bert")
tokenizer = AutoTokenizer.from_pretrained("adanish91/safety-qa-bert")

Example usage
context = "Employee was carrying 5-gallon fuel cans up top the warming heaters and strained shoulder."
question = "What body part was injured?"

inputs = tokenizer(question, context, return_tensors="pt")
outputs = model(**inputs)

answer_start = outputs.start_logits.argmax()
answer_end = outputs.end_logits.argmax()
answer = tokenizer.decode(inputs["input_ids"][answer_start:answer_end+1])
print(f"Answer: {answer}")

### Training

Load dataset from Hugging Face
from datasets import load_dataset

dataset = load_dataset("adanish91/safety-qa-bert-dataset")

Run training script
python train_safety_qa.py
--dataset_name adanish91/safety-qa-bert-dataset
--output_dir ./outputs
--num_train_epochs 50
--per_device_train_batch_size 8
--learning_rate 3e-6

## Dataset

The model is trained on two datasets available on Hugging Face:

- **seed_annotated_data.csv**: 149 manually annotated safety incident records
- **training_data.csv**: Extended dataset from active learning iterations

🔗 [Access Dataset on Hugging Face](https://huggingface.co/datasets/adanish91/safety-qa-bert-dataset)

### Data Format

Each record contains:
- `narrative`: Full text of safety incident
- `body_part`: Affected body parts
- `work_activity`: Activity being performed
- `accident_cause`: Cause/mechanism of accident

## Model Architecture

- **Base Model**: SafetyBERT (domain-adapted BERT for safety text)
- **Task**: Extractive Question Answering
- **Framework**: Hugging Face Transformers
- **Training**: Active learning with uncertainty sampling

## Training Details

### Hyperparameters

TrainingArguments(
per_device_train_batch_size=8,
per_device_eval_batch_size=8,
num_train_epochs=50,
learning_rate=3e-6,
weight_decay=0.01,
warmup_steps=200,
fp16=True, # Mixed precision training
evaluation_strategy="epoch",
save_strategy="epoch",
load_best_model_at_end=True,
metric_for_best_model="f1"
)

### Active Learning Process

1. Start with seed annotated data (149 examples)
2. Train initial model
3. Sample unlabeled narratives using uncertainty
4. Get annotations for selected samples
5. Retrain model with expanded dataset
6. Repeat until performance converges

## Resources

- **Model**: [adanish91/safety-qa-bert](https://huggingface.co/adanish91/safety-qa-bert)
- **Dataset**: [adanish91/safety-qa-bert-dataset](https://huggingface.co/datasets/adanish91/safety-qa-bert-dataset)
- **Paper**: [Link when published]

