#!/usr/bin/env python3
"""
SafetyBERT Self-Training QA System - Modified for Real Annotated Data
====================================================================
Uses real annotated data for initial training.
"""

import json
import random
import torch
import numpy as np
import pandas as pd
import os
from typing import List, Dict, Tuple, Optional
import re
from collections import defaultdict
from datetime import datetime

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

# Core imports
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    BertForQuestionAnswering, 
    TrainingArguments, 
    Trainer,
    default_data_collator,
    EarlyStoppingCallback
)
from torch.nn.functional import softmax
import torch.nn.functional as F

# Set seeds for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seeds(42)

# ============================================================================
# PHASE 1: Load Real Annotated Data & Base Model Training
# ============================================================================

def load_annotated_data(file_path="annotated_data_merged_reviewed.csv"):
    """Load real annotated data from CSV file"""
    
    print(f"Loading annotated data from {file_path}...")
    
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded CSV with {len(df)} rows and columns: {list(df.columns)}")
        
        # Check required columns
        required_columns = ['narrative', 'body_part', 'work_activity', 'accident_cause']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            return []
        
        # Clean data - remove rows with missing narratives
        df = df.dropna(subset=['narrative'])
        df = df[df['narrative'].str.strip() != '']
        
        print(f"After cleaning: {len(df)} rows with valid narratives")
        
        # Convert to list of dictionaries
        annotated_data = []
        for idx, row in df.iterrows():
            annotated_data.append({
                'id': f"annotated_{idx:04d}",
                'narrative': str(row['narrative']).strip(),
                'body_part': str(row['body_part']).strip() if pd.notna(row['body_part']) else '',
                'work_activity': str(row['work_activity']).strip() if pd.notna(row['work_activity']) else '',
                'accident_cause': str(row['accident_cause']).strip() if pd.notna(row['accident_cause']) else ''
            })
        
        print(f"Converted {len(annotated_data)} annotated samples")
        
        # Show statistics
        body_part_count = sum(1 for item in annotated_data if item['body_part'] and item['body_part'] != 'nan')
        work_activity_count = sum(1 for item in annotated_data if item['work_activity'] and item['work_activity'] != 'nan')
        accident_cause_count = sum(1 for item in annotated_data if item['accident_cause'] and item['accident_cause'] != 'nan')
        
        print(f"Answer statistics:")
        print(f"Body part answers: {body_part_count}/{len(annotated_data)} ({body_part_count/len(annotated_data)*100:.1f}%)")
        print(f"Work activity answers: {work_activity_count}/{len(annotated_data)} ({work_activity_count/len(annotated_data)*100:.1f}%)")
        print(f"Accident cause answers: {accident_cause_count}/{len(annotated_data)} ({accident_cause_count/len(annotated_data)*100:.1f}%)")
        
        return annotated_data
        
    except Exception as e:
        print(f"Error loading annotated data: {e}")
        return []


def convert_annotated_to_squad_format(annotated_data):
    """Convert annotated data to SQuAD v2.0 style QA format"""
    
    print("Converting annotated data to SQuAD v2.0 format...")
    
    # Updated questions to match your requirements
    questions = [
        "What body part was injured and what type of injury occurred?",
        "What specific work activity was the employee performing when the accident occurred?", 
        "What was the cause of accident?"
    ]
    
    answer_keys = ["body_part", "work_activity", "accident_cause"]
    
    squad_data = []
    stats = {"with_answer": 0, "no_answer": 0, "multi_span": 0}
    
    for item in annotated_data:
        narrative = item["narrative"]
        
        for question, answer_key in zip(questions, answer_keys):
            answer_text = item[answer_key]
            
            # Debug print for first few examples
            if len(squad_data) < 5:
                print(f"Debug - Question: {question}")
                print(f"Debug - Answer text: '{answer_text}'")
                print(f"Debug - In narrative: {answer_text in narrative if answer_text else False}")
            
            if answer_text and answer_text.strip() != '' and answer_text.lower() != 'nan':
                # Handle multiple answers (semicolon separated)
                if ';' in answer_text:
                    answer_texts = [ans.strip() for ans in answer_text.split(';') if ans.strip()]
                    answer_starts = []
                    valid_answers = []
                    
                    print(f"Processing {len(answer_texts)} spans for {answer_key}")
                    
                    for ans in answer_texts:
                        # Try exact match first
                        start_pos = narrative.find(ans)
                        if start_pos == -1:
                            # Try case-insensitive
                            start_pos = narrative.lower().find(ans.lower())
                            if start_pos != -1:
                                # Use actual text from narrative to preserve case
                                ans = narrative[start_pos:start_pos + len(ans)]
                                print(f"Case-corrected span: '{ans}'")
                        
                        if start_pos != -1:
                            # Verify the extraction
                            extracted = narrative[start_pos:start_pos + len(ans)]
                            if extracted != ans:
                                print(f"Extraction mismatch: expected '{ans}', got '{extracted}'")
                                ans = extracted  # Use what was actually extracted
                            
                            valid_answers.append(ans)
                            answer_starts.append(start_pos)
                            print(f"Found span: '{ans}' at position {start_pos}")
                        else:
                            print(f"WARNING: Span '{ans}' not found in narrative")
                
                    if valid_answers:
                        squad_entry = {
                            "id": f"{item['id']}_{answer_key}",
                            "context": narrative,
                            "question": question,
                            "question_type": answer_key,
                            "answers": {
                                "text": valid_answers,
                                "answer_start": answer_starts
                            },
                            "is_impossible": False
                        }
                        stats["with_answer"] += 1
                        stats["multi_span"] += 1
                        print(f"Created multi-span entry with {len(valid_answers)} spans")
                    else:
                        # No valid spans found, treat as no answer
                        squad_entry = {
                            "id": f"{item['id']}_{answer_key}",
                            "context": narrative,
                            "question": question,
                            "question_type": answer_key,
                            "answers": {
                                "text": [],
                                "answer_start": []
                            },
                            "is_impossible": True
                        }
                        stats["no_answer"] += 1
                        print(f"No valid spans found, marked as impossible")
                
                else:
                    # Single answer (existing logic)
                    if answer_text in narrative:
                        answer_start = narrative.find(answer_text)
                    else:
                        # Try case-insensitive
                        answer_start = narrative.lower().find(answer_text.lower())
                        if answer_start != -1:
                            # Use actual text from narrative to preserve case
                            actual_answer = narrative[answer_start:answer_start + len(answer_text)]
                            answer_text = actual_answer
                    
                    if answer_start != -1:
                        # Verify the extraction
                        extracted = narrative[answer_start:answer_start + len(answer_text)]
                        if extracted != answer_text:
                            print(f"Extraction mismatch: expected '{answer_text}', got '{extracted}'")
                        
                        squad_entry = {
                            "id": f"{item['id']}_{answer_key}",
                            "context": narrative,
                            "question": question,
                            "question_type": answer_key,
                            "answers": {
                                "text": [answer_text],
                                "answer_start": [answer_start]
                            },
                            "is_impossible": False
                        }
                        stats["with_answer"] += 1
                    else:
                        # No answer found
                        squad_entry = {
                            "id": f"{item['id']}_{answer_key}",
                            "context": narrative,
                            "question": question,
                            "question_type": answer_key,
                            "answers": {
                                "text": [],
                                "answer_start": []
                            },
                            "is_impossible": True
                        }
                        stats["no_answer"] += 1
            else:
                # No answer case (SQuAD v2.0 format)
                squad_entry = {
                    "id": f"{item['id']}_{answer_key}",
                    "context": narrative,
                    "question": question,
                    "question_type": answer_key,
                    "answers": {
                        "text": [],
                        "answer_start": []
                    },
                    "is_impossible": True
                }
                stats["no_answer"] += 1
            
            squad_data.append(squad_entry)
    
    print(f"Created {len(squad_data)} QA pairs (SQuAD v2.0 format)")
    print(f"With answers: {stats['with_answer']}")
    print(f"Multi-span answers: {stats['multi_span']}")
    print(f"No answers: {stats['no_answer']}")
    
    return squad_data

def create_train_val_test_split(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split data into train/validation/test sets"""
    
    print(f"\nCreating train/validation/test split...")
    
    # Set seed for reproducible splits
    random.seed(42)
    random.shuffle(data)
    
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    print(f"Train: {len(train_data)} samples ({len(train_data)/n:.1%})")
    print(f"Validation: {len(val_data)} samples ({len(val_data)/n:.1%})")
    print(f"Test: {len(test_data)} samples ({len(test_data)/n:.1%})")
    
    return train_data, val_data, test_data

def setup_model_and_tokenizer(model_name="bert-base-uncased"):
    """Setup BERT model and tokenizer for QA"""
    
    print(f"\nSetting up model and tokenizer: {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = BertForQuestionAnswering.from_pretrained(model_name)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        print(f"Model loaded on device: {device}")
        print(f"Tokenizer loaded with vocab size: {tokenizer.vocab_size}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def preprocess_data(data, tokenizer, max_length=128):
    """Preprocess QA data for training (SQuAD v2.0 compatible)"""
    
    def preprocess_function(examples):
        questions = [q.strip() for q in examples["question"]]
        contexts = examples["context"]
        
        # Tokenize
        tokenized_examples = tokenizer(
            questions,
            contexts,
            truncation="only_second",
            max_length=max_length,
            stride=64,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        
        # Handle the offset mapping and sample mapping
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")
        
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        
        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            
            sequence_ids = tokenized_examples.sequence_ids(i)
            
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            
            # Check if this is an impossible question (SQuAD v2.0)
            is_impossible = examples.get("is_impossible", [False] * len(examples["question"]))[sample_index]
            
            # For impossible questions or empty answers, point to CLS token
            if is_impossible or len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])
                
                # Find the start and end of the context in the tokenized input
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1
                    
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1
                    
                # Check if the answer is outside the context window
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    # Answer is outside context window, treat as impossible
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Find the tokens corresponding to the answer
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)
        
        return tokenized_examples
    
    print(f"\nPreprocessing {len(data)} samples...")
    
    dataset = Dataset.from_list(data)
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    print(f"Preprocessed {len(tokenized_dataset)} samples")
    return tokenized_dataset

def create_trainer(model, tokenizer, train_dataset, val_dataset):
    """Create trainer with early stopping"""
    
    print("\nSetting up training configuration...")
    
    training_args = TrainingArguments(
        output_dir="./results_base_model",
        eval_strategy="epoch",
        # eval_steps=25,
        save_strategy="epoch", 
        # save_steps=25,
        logging_steps=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=50,
        weight_decay=0.01,
        warmup_steps=200,
        learning_rate=3e-6,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,
        dataloader_pin_memory=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    return trainer

# ============================================================================
# PHASE 2: Unlabeled Data Preparation (Accident Narratives)
# ============================================================================

def load_accident_narratives(file_path="data-mill.csv", narrative_column="narrative"):
    """Load all accident narratives as a single prediction pool"""
    
    print(f"\nLoading accident narratives from {file_path}...")
    
    try:
        df = pd.read_csv(file_path)
        
        if narrative_column not in df.columns:
            print(f"Column '{narrative_column}' not found. Available columns: {list(df.columns)}")
            return []
        
        # Clean narratives
        narratives = df[narrative_column].dropna().astype(str).tolist()
        narratives = [n.strip() for n in narratives if len(n.strip()) > 10]  # Filter short narratives
        
        print(f"Loaded {len(narratives)} narratives as prediction pool")
        print(f"All narratives will be used for iterative self-training")
        
        return narratives
        
    except Exception as e:
        print(f"Error loading narratives: {e}")
        return []

def generate_questions_for_narratives(narratives):
    """Generate synthetic questions for accident narratives using updated question set"""
    
    print(f"\nGenerating questions for {len(narratives)} narratives...")
    
    # Updated safety-specific question templates to match your requirements
    question_templates = [
        "What body part was injured and what type of injury occurred?",
        "What specific work activity was the employee performing when the accident occurred?",
        "What was the cause of accident?"
    ]
    
    synthetic_qa_pairs = []
    
    for i, narrative in enumerate(narratives):
        # Generate multiple questions per narrative
        for template in question_templates:
            synthetic_qa_pairs.append({
                'id': f"synthetic_{i}_{len(synthetic_qa_pairs)}",
                'question': template,
                'context': narrative,
                'narrative_id': i,
                'is_synthetic': True,
                'answers': {
                    'text': [],
                    'answer_start': []
                },
                'is_impossible': True 
            })
    
    print(f"Generated {len(synthetic_qa_pairs)} synthetic QA pairs")
    print(f"Average {len(synthetic_qa_pairs)/len(narratives):.1f} questions per narrative")
    
    return synthetic_qa_pairs

# ============================================================================
# PHASE 3: Self-Training Implementation (UNCHANGED)
# ============================================================================

def calculate_qa_confidence(start_logits, end_logits, predicted_answer, tokenizer):
    """Calculate confidence score for QA prediction"""
    
    # Convert logits to probabilities
    start_probs = F.softmax(start_logits, dim=-1)
    end_probs = F.softmax(end_logits, dim=-1)
    
    # Get max probabilities
    max_start_prob = torch.max(start_probs).item()
    max_end_prob = torch.max(end_probs).item()
    
    # Combined probability confidence
    prob_confidence = (max_start_prob * max_end_prob) ** 0.5
    
    # Answer quality heuristics
    answer_tokens = tokenizer.tokenize(predicted_answer)
    
    # Length penalty (prefer 2-15 tokens)
    length_penalty = 1.0
    # if len(answer_tokens) < 2:
    #     length_penalty = 0.3
    # elif len(answer_tokens) > 15:
    #     length_penalty = 0.7
    
    # Generic answer penalty
    # generic_answers = ["no", "yes", "unclear", "unknown", "none", "n/a"]
    # generic_penalty = 0.2 if predicted_answer.lower().strip() in generic_answers else 1.0

    generic_penalty = 1
    
    # Empty answer penalty
    # empty_penalty = 0.1 if len(predicted_answer.strip()) == 0 else 1.0

    empty_penalty = 1
    
    
    # Combined confidence score
    final_confidence = prob_confidence * length_penalty * generic_penalty * empty_penalty
    
    return final_confidence

def predict_with_confidence(model, tokenizer, qa_pairs, batch_size=16):
    """Generate predictions with confidence scores"""
    
    print(f"\nGenerating predictions for {len(qa_pairs)} QA pairs...")
    
    model.eval()
    device = next(model.parameters()).device
    
    predictions = []
    
    for i in range(0, len(qa_pairs), batch_size):
        batch = qa_pairs[i:i+batch_size]
        
        # Prepare batch inputs
        questions = [item['question'] for item in batch]
        contexts = [item['context'] for item in batch]
        
        # Tokenize batch
        inputs = tokenizer(
            questions,
            contexts,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding=True,
            return_offsets_mapping=True
        )
        
        # EXTRACT offset_mapping BEFORE moving to device
        offset_mappings = inputs.pop("offset_mapping")
        
        # Now move the rest to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Process each item in batch
        for j, item in enumerate(batch):
            start_logits = outputs.start_logits[j]
            end_logits = outputs.end_logits[j]
            
            # Get predicted answer indices
            start_idx = torch.argmax(start_logits).item()
            end_idx = torch.argmax(end_logits).item()
            
            # Extract using offset mapping (EXACT from original text)
            if start_idx <= end_idx and start_idx < len(offset_mappings[j]) and end_idx < len(offset_mappings[j]):
                # Get character positions in original text
                start_char = offset_mappings[j][start_idx][0]
                end_char = offset_mappings[j][end_idx][1]
                
                # Extract EXACT text from original context
                predicted_answer = item['context'][start_char:end_char].strip()
            else:
                predicted_answer = ""
            
            # Calculate confidence
            confidence = calculate_qa_confidence(start_logits, end_logits, predicted_answer, tokenizer)
            
            predictions.append({
                'id': item['id'],
                'question': item['question'],
                'context': item['context'],
                'predicted_answer': predicted_answer,
                'confidence': confidence,
                'narrative_id': item.get('narrative_id', -1),
                'start_idx': start_idx, 
                'end_idx': end_idx,      
                'start_char': start_char if predicted_answer else -1,  # NEW: character positions
                'end_char': end_char if predicted_answer else -1       # NEW: character positions
            })
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"Processed {i + len(batch)}/{len(qa_pairs)} samples...")
    
    print(f"Generated {len(predictions)} predictions")
    return predictions

def get_question_type(question):
    """Determine question type from question text - UPDATED for new questions"""
    question_lower = question.lower().strip()
    
    if "body part" in question_lower and "injury" in question_lower:
        return "body_part"
    elif "work activity" in question_lower or "performing" in question_lower:
        return "work_activity"
    elif "cause" in question_lower and "accident" in question_lower:
        return "accident_cause"
    else:
        # Fallback classification
        return "unknown"

def select_hybrid_samples(all_predictions, used_indices, max_samples=200):
    """
    Hybrid selection: N/2 low confidence + N/2 high confidence samples
    """
    print(f"Selecting samples with hybrid strategy (low + high confidence)...")
    print(f"Already used: {len(used_indices)} predictions")
    print(f"Target samples: {max_samples} (split: {max_samples//2} low + {max_samples - max_samples//2} high)")
    
    # Filter out used predictions and load previous reviews
    previous_reviews = load_previous_reviews()
    
    available_predictions = []
    for i, pred in enumerate(all_predictions):
        original_idx = pred.get('original_index', i)
        if original_idx not in used_indices:
            if len(pred['predicted_answer'].strip()) > 0:  # Has answer
                
                # Check if this sample is already in training sets
                sample_key = create_sample_key(pred['context'], pred['question'])
                if sample_key in previous_reviews:
                    review_data = previous_reviews[sample_key]
                    if review_data.get('your_decision') == 'rejected':
                        continue  # Skip rejected samples
                    if review_data.get('reused', False) == True:
                        continue  # Skip already used samples
                
                available_predictions.append((original_idx, pred))
    
    print(f"Available predictions after filtering: {len(available_predictions)}")
    
    if len(available_predictions) == 0:
        print("No available predictions!")
        return []
    
    # Group by question type for balanced selection
    by_question_type = defaultdict(list)
    for idx, pred in available_predictions:
        question_type = get_question_type(pred['question'])
        if question_type != "unknown":
            by_question_type[question_type].append((idx, pred))
    
    if len(by_question_type) == 0:
        print("No valid question types found!")
        return []
    
    # Calculate samples per question type
    available_types = list(by_question_type.keys())
    samples_per_type = max_samples // len(available_types)
    low_per_type = samples_per_type // 2
    high_per_type = samples_per_type - low_per_type
    
    print(f"\nHybrid Strategy per question type:")
    print(f"Total per type: {samples_per_type} ({low_per_type} low + {high_per_type} high confidence)")
    
    selected_samples = []
    
    for question_type in sorted(available_types):
        candidates = by_question_type[question_type]
        
        if len(candidates) == 0:
            continue
            
        # Sort by confidence for selection
        candidates.sort(key=lambda x: x[1]['confidence'])
        
        # Apply diversity constraint
        narrative_counts = defaultdict(int)
        max_per_narrative = 3  # Reduced since we're splitting between low/high
        
        # Select LOW confidence samples (from start of sorted list)
        low_conf_selected = []
        for idx, pred in candidates:
            if len(low_conf_selected) >= low_per_type:
                break
            narrative_id = pred['narrative_id']
            if narrative_counts[narrative_id] >= max_per_narrative:
                continue
            low_conf_selected.append(pred)
            narrative_counts[narrative_id] += 1
        
        # Reset narrative counts for high confidence selection
        narrative_counts = defaultdict(int)
        
        # Select HIGH confidence samples (from end of sorted list)
        high_conf_selected = []
        for idx, pred in reversed(candidates):  # Start from highest confidence
            if len(high_conf_selected) >= high_per_type:
                break
            narrative_id = pred['narrative_id']
            if narrative_counts[narrative_id] >= max_per_narrative:
                continue
            # Only select high confidence samples above minimum threshold
            if pred['confidence'] >= 0.6:  # Minimum threshold for "high confidence"
                high_conf_selected.append(pred)
                narrative_counts[narrative_id] += 1
        
        # Add selection info
        if low_conf_selected or high_conf_selected:
            all_low_confs = [s['confidence'] for s in low_conf_selected]
            all_high_confs = [s['confidence'] for s in high_conf_selected]
            
            print(f"{question_type}:")
            if low_conf_selected:
                print(f"  Low confidence: {len(low_conf_selected)} samples (range: {min(all_low_confs):.3f} - {max(all_low_confs):.3f})")
            if high_conf_selected:
                print(f"  High confidence: {len(high_conf_selected)} samples (range: {min(all_high_confs):.3f} - {max(all_high_confs):.3f})")
            
            # Add confidence type flag for review process
            for sample in low_conf_selected:
                sample['selection_type'] = 'low_confidence'
            for sample in high_conf_selected:
                sample['selection_type'] = 'high_confidence'
            
            selected_samples.extend(low_conf_selected + high_conf_selected)
        else:
            print(f"{question_type}: No samples selected")
    
    print(f"\nTotal selected: {len(selected_samples)} samples")
    
    # Show overall statistics
    if selected_samples:
        low_conf_total = sum(1 for s in selected_samples if s.get('selection_type') == 'low_confidence')
        high_conf_total = sum(1 for s in selected_samples if s.get('selection_type') == 'high_confidence')
        
        print(f"Final breakdown: {low_conf_total} low confidence + {high_conf_total} high confidence")
        
        all_confidences = [s['confidence'] for s in selected_samples]
        print(f"Overall confidence range: {min(all_confidences):.3f} - {max(all_confidences):.3f}")
    
    return selected_samples

    
def convert_to_training_format(selected_predictions):
    """Convert high-confidence predictions to training format"""
    
    training_samples = []
    
    for pred in selected_predictions:
        training_samples.append({
            'id': pred['id'],
            'question': pred['question'],
            'context': pred['context'],
            'answers': {
                'text': [pred['predicted_answer']],
                'answer_start': [pred.get('start_idx', 0)]  # Approximate
            },
            'is_pseudo_labeled': True,
            'confidence': pred['confidence']
        })
    
    return training_samples

def evaluate_model_qa(model, tokenizer, test_data, sample_name="Test Set", print_wrong_predictions=False):
    """Complete evaluation with F1, Exact Match, and Fuzzy Match metrics"""
    
    import re
    import string
    from collections import Counter
    from difflib import SequenceMatcher
    
    print(f"\nEvaluating {sample_name} with Complete Metrics...")
    
    def normalize_answer(s, context=""):
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def fix_tokenization_spacing(text):
            # Fix common tokenization issues
            text = re.sub(r"(\w)\s+\'\s*([a-z])", r"\1'\2", text) 
            text = re.sub(r"\s+\'\s*s\b", "'s", text)              
            return text
        def lower(text):
            return text.lower()
        
        # Fix spacing FIRST, then normalize normally (keeping apostrophes!)
        s = fix_tokenization_spacing(s)
        return white_space_fix(remove_articles(lower(s)))
    
    def compute_f1(prediction, ground_truth, context=""):
        pred_tokens = normalize_answer(prediction, context).split()
        truth_tokens = normalize_answer(ground_truth, context).split()
        
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return int(pred_tokens == truth_tokens)
        
        common_tokens = Counter(pred_tokens) & Counter(truth_tokens)
        num_common = sum(common_tokens.values())
        
        if num_common == 0:
            return 0
            
        precision = num_common / len(pred_tokens)
        recall = num_common / len(truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    
    def compute_exact_match(prediction, ground_truth, context=""):
        return normalize_answer(prediction, context) == normalize_answer(ground_truth, context)
    
    def compute_fuzzy_match(prediction, ground_truth, context="", threshold=0.6):
        """Compute fuzzy string similarity using SequenceMatcher"""
        pred_norm = normalize_answer(prediction, context)
        truth_norm = normalize_answer(ground_truth, context)
        
        if not pred_norm and not truth_norm:
            return 1.0
        if not pred_norm or not truth_norm:
            return 0.0
            
        similarity = SequenceMatcher(None, pred_norm, truth_norm).ratio()
        return 1.0 if similarity >= threshold else 0.0
    
    model.eval()
    device = next(model.parameters()).device
    
    exact_matches = []
    f1_scores = []
    fuzzy_matches = []
    wrong_predictions = []  # Store wrong predictions
    
    for i, example in enumerate(test_data):
        inputs = tokenizer(
            example["question"],
            example["context"], 
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding=True
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        start_idx = torch.argmax(outputs.start_logits[0]).item()
        end_idx = torch.argmax(outputs.end_logits[0]).item()
        
        if start_idx <= end_idx and start_idx < len(inputs["input_ids"][0]) and end_idx < len(inputs["input_ids"][0]):
            predicted_answer = tokenizer.decode(
                inputs["input_ids"][0][start_idx:end_idx+1],
                skip_special_tokens=True
            ).strip()
        else:
            predicted_answer = ""
        
        if example["answers"]["text"]:
            true_answers = example["answers"]["text"] if isinstance(example["answers"]["text"], list) else [example["answers"]["text"]]
        else:
            true_answers = [""]
        
        # Get context for evaluation
        context = example.get('context', '')
        
        # Compute best scores across all possible answers
        max_exact = 0
        max_f1 = 0
        max_fuzzy = 0
        best_true_answer = true_answers[0]  # Keep track of best matching true answer
        
        for true_answer in true_answers:
            exact = compute_exact_match(predicted_answer, true_answer, context)
            f1 = compute_f1(predicted_answer, true_answer, context)
            fuzzy = compute_fuzzy_match(predicted_answer, true_answer, context)
            
            # Update best scores and keep track of best true answer
            if f1 > max_f1:
                max_f1 = f1
                best_true_answer = true_answer
            if exact > max_exact:
                max_exact = exact
            if fuzzy > max_fuzzy:
                max_fuzzy = fuzzy
        
        exact_matches.append(max_exact)
        f1_scores.append(max_f1)
        fuzzy_matches.append(max_fuzzy)
        
        # **COLLECT WRONG PREDICTIONS**
        if max_exact == 0:  # If exact match is 0, it's wrong
            wrong_predictions.append({
                'question': example['question'],
                'context': example['context'],
                'true_answer': best_true_answer,
                'predicted_answer': predicted_answer,
                'f1_score': max_f1,
                'fuzzy_score': max_fuzzy,
                'example_id': example.get('id', f'sample_{i}')
            })
        
        # Show detailed examples for first few samples
        if i < 5:
            similarity_score = SequenceMatcher(None, normalize_answer(predicted_answer, context), normalize_answer(true_answers[0], context)).ratio()
            status = "CORRECT" if max_exact == 1 else "WRONG"
            print(f"Sample {i+1} ({status}): Q: {example['question']}...")
            
            context_display = example.get('context', 'No context available')
            if len(context_display) > 200:
                print(f" Context: {context_display[:200]}...")
                print(f"[...truncated from {len(context_display)} characters]")
            else:
                print(f"Context: {context_display}")
            
            print(f"Ground Truth: '{true_answers[0]}'")
            print(f"Prediction:   '{predicted_answer}'")
            print(f"EM: {max_exact} | F1: {max_f1:.3f} | Fuzzy: {max_fuzzy} | Sim: {similarity_score:.3f}")
            print()
    
    # Calculate final scores
    em_score = np.mean(exact_matches) * 100
    f1_score = np.mean(f1_scores) * 100
    fuzzy_score = np.mean(fuzzy_matches) * 100
    
    print(f"{sample_name} Complete Results:")
    print(f"F1 Score: {f1_score:.1f}%")
    print(f"Exact Match: {em_score:.1f}%") 
    print(f"Fuzzy Match: {fuzzy_score:.1f}%")
    print(f"Total Questions: {len(test_data)}")
    print(f"Wrong Predictions: {len(wrong_predictions)}")
    
    # **PRINT WRONG PREDICTIONS IF REQUESTED**
    if print_wrong_predictions and len(wrong_predictions) > 0:
        print(f"\nWRONG PREDICTIONS FOR {sample_name}")
        print("=" * 80)
        
        for i, wrong in enumerate(wrong_predictions):
            print(f"\nWrong Prediction #{i+1} (ID: {wrong['example_id']})")
            print("-" * 60)
            
            # Print context (truncate if too long)
            context = wrong['context']
            if len(context) > 300:
                print(f"CONTEXT: {context[:300]}...")
                print(f"[...truncated from {len(context)} characters]")
            else:
                print(f"CONTEXT: {context}")
            
            print(f"QUESTION: {wrong['question']}")
            print(f"TRUE ANSWER: '{wrong['true_answer']}'")
            print(f"PREDICTED: '{wrong['predicted_answer']}'")
            print(f"F1: {wrong['f1_score']:.3f} | Fuzzy: {wrong['fuzzy_score']:.3f}")
            
            # Add separator between examples
            if i < len(wrong_predictions) - 1:
                print()
        
        print("=" * 80)
        print(f"Showed {len(wrong_predictions)} wrong predictions out of {len(test_data)} total")
    
    # Return F1 as primary metric
    return f1_score


def create_sample_key(narrative, question):
    """Create a unique key for narrative + question combination"""
    narrative_snippet = narrative.strip()
    return f"{narrative_snippet}|||{question.strip()}"

def load_previous_reviews(review_file="reviewed_samples.json"):
    """Load previous reviews and create lookup dictionary"""
    if not os.path.exists(review_file):
        return {}
    
    try:
        with open(review_file, 'r') as f:
            reviews = json.load(f)
        
        # Create lookup dictionary: sample_key -> review_data
        lookup = {}
        for review_key, review_data in reviews.items():
            sample_key = create_sample_key(review_data['narrative'], review_data['question'])
            lookup[sample_key] = review_data
        
        return lookup
    except Exception as e:
        print(f"Error loading previous reviews: {e}")
        return {}

def save_single_review(review_entry, iteration, sample_index, review_file="reviewed_samples.json"):
    """Save a single review immediately to prevent data loss"""
    try:
        # Load existing reviews
        if os.path.exists(review_file):
            with open(review_file, 'r') as f:
                existing_reviews = json.load(f)
        else:
            existing_reviews = {}
        
        # Create unique key for this review
        review_key = f"iter_{iteration}_sample_{sample_index}_{datetime.now().strftime('%H%M%S')}"
        existing_reviews[review_key] = review_entry
        
        # Save immediately
        with open(review_file, 'w') as f:
            json.dump(existing_reviews, f, indent=2)
        
        print(f"Review saved: {review_key}")
        return True
        
    except Exception as e:
        print(f"Error saving review: {e}")
        return False

def update_existing_review_reused_status(sample_key, iteration, review_file="reviewed_samples.json"):
    """Update existing review entry's reused status without creating duplicates"""
    try:
        if os.path.exists(review_file):
            with open(review_file, 'r') as f:
                all_reviews = json.load(f)
            
            # Find and update the existing entry
            for key, review in all_reviews.items():
                existing_key = create_sample_key(review['narrative'], review['question'])
                if existing_key == sample_key:
                    all_reviews[key]['reused'] = True
                    all_reviews[key]['last_used_iteration'] = iteration
                    # print(f"Updated existing review: {key}")
                    break
            
            # Save back
            with open(review_file, 'w') as f:
                json.dump(all_reviews, f, indent=2)
            
    except Exception as e:
        print(f"Error updating review: {e}")

def interactive_review(hybrid_samples, iteration, all_synthetic_qa_pairs):
    """
    Interactive review with reuse analysis per question type
    Returns: (approved_samples, all_processed_indices)
    """
    print(f"\nINTERACTIVE REVIEW - Iteration {iteration}")
    print(f"Processing {len(hybrid_samples)} pre-selected samples")
    
    # Load previous reviews
    previous_reviews = load_previous_reviews()
    print(f"Loaded {len(previous_reviews)} previous reviews for reuse")
    
    # Group hybrid_samples by question type
    samples_by_type = defaultdict(list)
    for sample in hybrid_samples:
        question_type = get_question_type(sample['question'])
        samples_by_type[question_type].append(sample)
    
    print(f"\nPre-selected samples by question type:")
    for question_type, samples in samples_by_type.items():
        if len(samples) > 0:
            confidences = [s['confidence'] for s in samples]
            print(f"{question_type}: {len(samples)} samples "
                  f"(confidence: {max(confidences):.2f} - {min(confidences):.2f})")
    
    # ============================================================================
    # STEP 2: ANALYZE REUSED VS NEW FOR EACH QUESTION TYPE
    # ============================================================================
    print(f"\nREUSE ANALYSIS:")
    print("=" * 25)
    
    approved_samples = []
    review_plan = {}
    total_reused = 0
    total_to_review = 0
    total_reviewed = 0
    reused_original_indices = []  # Track original indices for reused samples
    new_reviewed_indices = []     # Track original indices for newly reviewed samples
    
    for question_type, samples in samples_by_type.items():
        if len(samples) == 0:
            continue
            
        # Check which samples are already reviewed
        reused_samples = []
        new_samples = []
        
        for sample in samples:
            sample_key = create_sample_key(sample['context'], sample['question'])
            
            if sample_key in previous_reviews:
                previous_review = previous_reviews[sample_key]
                # Only reuse if previously accepted/corrected
                if previous_review['your_decision'] in ['accepted', 'corrected']:
                    reused_samples.append((sample, previous_review))
                elif previous_review['your_decision'] == 'rejected':
                    continue  # Skip rejected samples
                else:
                    new_samples.append(sample)  # Unknown decision, treat as new
            else:
                new_samples.append(sample)
        
        # Store plan for this question type
        review_plan[question_type] = {
            'target': len(samples),
            'reused': reused_samples,
            'new': new_samples,
            'reused_count': len(reused_samples),
            'new_count': len(new_samples)
        }
        
        total_reused += len(reused_samples)
        total_to_review += len(new_samples)
    
    print(f"\nTOTAL: {total_reused} reused + {total_to_review} new = {total_reused + total_to_review} samples")
    
    # ============================================================================
    # PROCESS REUSED SAMPLES (AUTOMATIC) + TRACK ORIGINAL INDICES
    # ============================================================================
    if total_reused > 0:
        print(f"\nProcessing {total_reused} reused samples...")
        print("-" * 40)
        
        reused_count = 0
        for question_type, plan in review_plan.items():
            for sample, previous_review in plan['reused']:
                reused_count += 1
                sample_key = create_sample_key(sample['context'], sample['question'])
                
                decision = previous_review['your_decision']
                
                if decision == 'accepted':
                    approved_samples.append(sample)
                    
                elif decision == 'corrected':
                    corrected_sample = sample.copy()
                    corrected_sample['predicted_answer'] = previous_review['final_answer']
                    corrected_sample['original_answer'] = sample['predicted_answer']
                    corrected_sample['human_corrected'] = True
                    corrected_sample['reused_correction'] = True
                    
                    if previous_review['final_answer'] == "":
                        corrected_sample['is_no_answer'] = True
                    
                    approved_samples.append(corrected_sample)
                
                # Find and track original index for this reused sample
                for i, original_qa in enumerate(all_synthetic_qa_pairs):
                    original_key = create_sample_key(original_qa['context'], original_qa['question'])
                    if original_key == sample_key:
                        reused_original_indices.append(i)
                        break
                
                # Update reused status (only for actually reused samples)
                update_existing_review_reused_status(sample_key, iteration)
    
    # ============================================================================
    # REVIEW NEW SAMPLES BY QUESTION TYPE
    # ============================================================================
    if total_to_review > 0:
        print(f"\nReviewing {total_to_review} new samples by question type...")
        print("=" * 60)
        print("Options:")
        print("(a)ccept - Use model's answer as-is")
        print("(c)orrect - Provide correct answer (or press Enter for 'no answer')")
        print("(r)eject - Don't use this sample for training")
        print("(s)kip - Skip remaining samples and continue")
        print("(q)uit - Quit and save progress")
        print("=" * 60)        
        
        for question_type in sorted(review_plan.keys()):
            plan = review_plan[question_type]
            new_samples = plan['new']
            
            if len(new_samples) == 0:
                continue
                
            print(f"\n=== REVIEWING {question_type.upper()} ({len(new_samples)} samples) ===")
            
            # Sort new samples by confidence (lowest first for review)
            new_samples.sort(key=lambda x: x['confidence'], reverse=False)
            
            type_reviewed = 0
            for sample in new_samples:
                total_reviewed += 1
                type_reviewed += 1
                
                print(f"\nSample {type_reviewed}/{len(new_samples)} for {question_type}")
                print(f"Global progress: {total_reviewed}/{total_to_review} | Confidence: {sample['confidence']:.3f}")
                print("=" * 80)
                
                # Display context
                narrative = sample['context']
                if len(narrative) > 500:
                    print(f"CONTEXT: {narrative[:500]}...")
                    print(f"[...truncated from {len(narrative)} characters]")
                else:
                    print(f"CONTEXT: {narrative}")
                
                print(f"\nQUESTION: {sample['question']}")
                print(f"MODEL ANSWER: \"{sample['predicted_answer']}\"")
                print(f"CONFIDENCE: {sample['confidence']:.3f}")
                
                # Get user input
                while True:
                    choice = input(f"\nChoice (a/c/r/s/q): ").lower().strip()
                    
                    if choice == 'a':  # Accept
                        approved_samples.append(sample)
                        print("? Accepted")
                        decision = "accepted"
                        final_answer = sample['predicted_answer']
                        
                        # NEW: Track original index for newly reviewed sample
                        sample_key = create_sample_key(sample['context'], sample['question'])
                        for i, original_qa in enumerate(all_synthetic_qa_pairs):
                            original_key = create_sample_key(original_qa['context'], original_qa['question'])
                            if original_key == sample_key:
                                new_reviewed_indices.append(i)
                                break
                        break
                        
                    elif choice == 'c':  # Correct
                        print("\nEnter correct answer (or press Enter for 'no answer'):")
                        corrected_answer = input("Correct answer: ").strip()
                        
                        corrected_sample = sample.copy()
                        corrected_sample['predicted_answer'] = corrected_answer
                        corrected_sample['original_answer'] = sample['predicted_answer']
                        corrected_sample['human_corrected'] = True
                        
                        if corrected_answer == "":
                            corrected_sample['is_no_answer'] = True
                            print("? Corrected to: NO ANSWER")
                        else:
                            corrected_sample['is_no_answer'] = False
                            print(f"? Corrected to: \"{corrected_answer}\"")
                        
                        approved_samples.append(corrected_sample)
                        decision = "corrected"
                        final_answer = corrected_answer
                        
                        # NEW: Track original index for newly reviewed sample
                        sample_key = create_sample_key(sample['context'], sample['question'])
                        for i, original_qa in enumerate(all_synthetic_qa_pairs):
                            original_key = create_sample_key(original_qa['context'], original_qa['question'])
                            if original_key == sample_key:
                                new_reviewed_indices.append(i)
                                break
                        break
                        
                    elif choice == 'r':  # Reject
                        print("? Rejected")
                        decision = "rejected"
                        final_answer = ""
                        
                        # NEW: Still track rejected samples as "used" so they won't be selected again
                        sample_key = create_sample_key(sample['context'], sample['question'])
                        for i, original_qa in enumerate(all_synthetic_qa_pairs):
                            original_key = create_sample_key(original_qa['context'], original_qa['question'])
                            if original_key == sample_key:
                                new_reviewed_indices.append(i)
                                break
                        break
                        
                    elif choice == 's':  # Skip remaining
                        print(f"Skipping remaining samples.")
                        print_current_summary(approved_samples, total_reused, total_reviewed, total_to_review)
                        # Return ALL processed indices (reused + newly reviewed so far)
                        return approved_samples, reused_original_indices + new_reviewed_indices
                        
                    elif choice == 'q':  # Quit
                        print("Quitting review process")
                        continue_choice = input("Continue training with current samples (y/n): ").lower().strip()
                        if continue_choice == 'y':
                            # Return ALL processed indices (reused + newly reviewed so far)
                            return approved_samples, reused_original_indices + new_reviewed_indices
                        else:
                            return [], []
                            
                    else:
                        print("Invalid choice. Please enter 'a', 'c', 'r', 's', or 'q'")
                
                # Save review immediately for new samples
                sample_key = create_sample_key(sample['context'], sample['question'])
                reused_value = decision in ['accepted', 'corrected']  # True if approved, False if rejected
                
                review_entry = {
                    "narrative": narrative,
                    "question": sample['question'],
                    "model_answer": sample['predicted_answer'],
                    "your_decision": decision,
                    "final_answer": final_answer,
                    "confidence": sample['confidence'],
                    "iteration": iteration,
                    "timestamp": datetime.now().isoformat(),
                    "reused": reused_value,  # Set to True for approved samples, False for rejected
                    "question_type": question_type
                }
                save_single_review(review_entry, iteration, total_reviewed)
    
    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    print(f"\nFINAL PROCESSING SUMMARY")
    print("=" * 50)
    print(f"Total reused samples processed: {total_reused}")
    print(f"Total new samples reviewed: {total_reviewed}")
    print(f"Total approved samples for training: {len(approved_samples)}")
    print(f"Original indices for reused samples: {len(reused_original_indices)}")
    print(f"Original indices for new reviewed samples: {len(new_reviewed_indices)}")
    
    # Show breakdown of approved samples
    reused_approved = sum(1 for s in approved_samples if s.get('reused_correction', False) or 'human_corrected' not in s)
    new_approved = len(approved_samples) - reused_approved
    
    print(f"Approved samples breakdown:")
    print(f"From reused: {reused_approved}")
    print(f"From new: {new_approved}")
    print(f"Total: {len(approved_samples)}")
    
    all_processed_indices = reused_original_indices + new_reviewed_indices
    if all_processed_indices:
        print(f"All processed indices (reused + new): {len(all_processed_indices)}")
    
    # Final summary
    print_current_summary(approved_samples, total_reused, total_reviewed, total_to_review)
    
    # Return approved samples AND ALL processed indices (both reused and newly reviewed)
    return approved_samples, all_processed_indices

def convert_reviewed_samples_to_training_format(reviewed_samples):
    """
    Convert reviewed samples to proper training format with answer_text and answer_start
    """
    converted_samples = []
    
    for sample in reviewed_samples:
        # Get the final answer (could be corrected or original)
        if sample.get('human_corrected', False):
            final_answer = sample['predicted_answer']  # This is the corrected answer
        else:
            final_answer = sample['predicted_answer']  # This is the accepted answer
        
        # Create training format sample
        training_sample = {
            'id': sample.get('id', 'unknown'),
            'context': sample['context'],
            'question': sample['question'],
            'answer_text': '',
            'answer_start': -1,
            'confidence': sample.get('confidence', 1.0), 
            'human_corrected': sample.get('human_corrected', False),  
            'original_answer': sample.get('original_answer', ''),  
        }
        
        # Calculate answer_start index
        if final_answer and final_answer.strip() != '':
            context = sample['context']
            
            # Try exact match first (case-sensitive)
            answer_start = context.find(final_answer)
            
            if answer_start != -1:
                # Perfect match - use as-is
                training_sample['answer_start'] = answer_start
                training_sample['answer_text'] = final_answer
            else:
                # Try case-insensitive search
                answer_start = context.lower().find(final_answer.lower())
                if answer_start != -1:
                    # Extract exact text from context to preserve original case
                    actual_text_from_context = context[answer_start:answer_start + len(final_answer)]
                    training_sample['answer_start'] = answer_start
                    training_sample['answer_text'] = actual_text_from_context  # Use context's exact case!
                    print(f"Case-corrected: '{final_answer}' '{actual_text_from_context}'")
                else:
                    # Try fuzzy matching for minor differences
                    import re
                    # Remove extra spaces and punctuation for matching
                    clean_answer = re.sub(r'[^\w\s]', '', final_answer.strip())
                    
                    # Search for similar text (allowing for minor variations)
                    pattern = re.escape(clean_answer).replace(r'\ ', r'\s+')
                    match = re.search(pattern, context, re.IGNORECASE)
                    
                    if match:
                        training_sample['answer_start'] = match.start()
                        training_sample['answer_text'] = context[match.start():match.end()]
                        print(f"Fuzzy-matched: '{final_answer}' '{training_sample['answer_text']}'")
                    else:
                        print(f"WARNING: Answer '{final_answer}' not found in context for {sample.get('id', 'unknown')}")
                        training_sample['answer_start'] = -1
                        training_sample['answer_text'] = ''  # Mark as no answer
        else:
            # No answer case
            training_sample['answer_text'] = ''
            training_sample['answer_start'] = -1
        
        converted_samples.append(training_sample)
    
    return converted_samples

def print_current_summary(approved_samples, reused_count, reviewed_count, total_new):
    """Print current review summary"""
    print(f"\nCURRENT REVIEW SUMMARY")
    print("=" * 40)
    print(f"Reused previous decisions: {reused_count}")
    print(f"Manually reviewed: {reviewed_count}/{total_new}")
    
    if approved_samples:
        no_answer_approved = sum(1 for sample in approved_samples if sample.get('is_no_answer', False))
        with_answer_approved = len(approved_samples) - no_answer_approved
        print(f"Total approved: {len(approved_samples)}")
        print(f"- With answers: {with_answer_approved}")
        print(f"- No answer (empty): {no_answer_approved}")
    else:
        print("Total approved: 0")   


def convert_to_training_format(selected_predictions):
    """Convert high-confidence predictions to training format with question type tracking"""
    
    training_samples = []
    
    for pred in selected_predictions:
        # Check if this is a no-answer case
        is_no_answer = pred.get('is_no_answer', False)
        predicted_answer = pred.get('answer_text', '')
        context = pred['context']
        question_type = get_question_type(pred['question'])  # Add question type tracking
        
        if is_no_answer or predicted_answer == "":
            # No answer case - SQuAD v2.0 format
            training_sample = {
                'id': pred['id'],
                'question': pred['question'],
                'context': context,
                'answers': {
                    'text': [],           # Empty list for no answer
                    'answer_start': []    # Empty list for no answer
                },
                'is_impossible': True,    # SQuAD v2.0 flag for no answer
                'is_pseudo_labeled': True,
                'confidence': pred['confidence'],
                'human_reviewed': pred.get('human_corrected', False),
                'question_type': question_type  # Add question type
            }
        else:
            # Has answer case - need to find correct position in context
            answer_start = pred.get('answer_start', -1)
            
            # If answer_start is missing or invalid, find it properly
            if answer_start == -1:
                # Try exact match first (case-sensitive)
                answer_start = context.find(predicted_answer)
                
                if answer_start == -1:
                    # Try case-insensitive search
                    answer_start = context.lower().find(predicted_answer.lower())
                    if answer_start != -1:
                        # Extract actual text from context to preserve case
                        actual_answer = context[answer_start:answer_start + len(predicted_answer)]
                        predicted_answer = actual_answer  # Use context's exact case
                        print(f"Training format case-corrected: '{pred.get('answer_text', '')}' '{actual_answer}'")
                    else:
                        # Try fuzzy matching
                        import re
                        clean_answer = re.sub(r'[^\w\s]', '', predicted_answer.strip())
                        pattern = re.escape(clean_answer).replace(r'\ ', r'\s+')
                        
                        match = re.search(pattern, context, re.IGNORECASE)
                        if match:
                            answer_start = match.start()
                            predicted_answer = context[match.start():match.end()]
                            print(f"Training format fuzzy-matched: '{pred.get('answer_text', '')}' '{predicted_answer}'")
                        else:
                            print(f"Warning: Answer '{predicted_answer}' not found in context, treating as no-answer")
                            # Convert to no-answer case
                            training_sample = {
                                'id': pred['id'],
                                'question': pred['question'],
                                'context': context,
                                'answers': {
                                    'text': [],
                                    'answer_start': []
                                },
                                'is_impossible': True,
                                'is_pseudo_labeled': True,
                                'confidence': pred['confidence'],
                                'human_reviewed': pred.get('human_corrected', False),
                                'question_type': question_type
                            }
                            training_samples.append(training_sample)
                            continue
            
            # Validate the answer position
            if answer_start >= 0 and answer_start + len(predicted_answer) <= len(context):
                text_at_position = context[answer_start:answer_start + len(predicted_answer)]
                if text_at_position != predicted_answer:
                    print(f"Position mismatch for {pred['id']}: expected '{predicted_answer}' but found '{text_at_position}' at position {answer_start}")
                    predicted_answer = text_at_position
            
            training_sample = {
                'id': pred['id'],
                'question': pred['question'],
                'context': context,
                'answers': {
                    'text': [predicted_answer],
                    'answer_start': [max(0, answer_start)]  # Ensure non-negative
                },
                'is_impossible': False,
                'is_pseudo_labeled': True,
                'confidence': pred['confidence'],
                'human_reviewed': pred.get('human_corrected', False),
                'question_type': question_type  # Add question type
            }
        
        training_samples.append(training_sample)
    
    # Print final distribution for verification
    distribution = defaultdict(int)
    for sample in training_samples:
        distribution[sample['question_type']] += 1
    
    print(f"Training format distribution: {dict(distribution)}")
    
    return training_samples

def display_review_statistics(review_file="reviewed_samples.json"):
    """Display statistics from all reviews with better error handling"""
    if not os.path.exists(review_file):
        print("No review file found")
        return
    
    try:
        with open(review_file, 'r') as f:
            reviews = json.load(f)
        
        print(f"\nCUMULATIVE REVIEW STATISTICS")
        print("=" * 40)
        print(f"Total entries in file: {len(reviews)}")
        
        # Filter out control entries (skip, quit, etc.)
        actual_reviews = {k: v for k, v in reviews.items() 
                         if v.get('your_decision') not in ['skipped_remaining', 'quit']}
        
        print(f"Actual sample reviews: {len(actual_reviews)}")
        
        if len(actual_reviews) > 0:
            decisions = [review['your_decision'] for review in actual_reviews.values()]
            print(f"Accepted: {decisions.count('accepted')}")
            print(f"Corrected: {decisions.count('corrected')}")
            print(f"Rejected: {decisions.count('rejected')}")
            
            # Group by iteration
            iterations = {}
            for review in actual_reviews.values():
                iter_num = review.get('iteration', 0)
                if iter_num not in iterations:
                    iterations[iter_num] = []
                iterations[iter_num].append(review['your_decision'])
            
            print("\nBy Iteration:")
            for iter_num in sorted(iterations.keys()):
                decisions = iterations[iter_num]
                print(f"Iteration {iter_num}: {len(decisions)} samples")
                print(f"Accepted: {decisions.count('accepted')}")
                print(f"Corrected: {decisions.count('corrected')}")
                print(f"Rejected: {decisions.count('rejected')}")
        
    except Exception as e:
        print(f"Error reading review statistics: {e}")

def split_approved_samples(approved_samples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Stratified split that maintains question type balance across train/val/test
    """
    if len(approved_samples) == 0:
        return [], [], []
    
    print(f"Performing stratified split of {len(approved_samples)} samples...")
    
    # Group samples by question type
    by_question_type = defaultdict(list)
    for sample in approved_samples:
        question_type = get_question_type(sample['question'])
        by_question_type[question_type].append(sample)
    
    print("Input distribution:")
    for q_type, samples in by_question_type.items():
        print(f"{q_type}: {len(samples)} samples")
    
    # Perform stratified split for each question type
    train_split = []
    val_split = []
    test_split = []
    
    random.seed(42)  # For reproducible splits
    
    for question_type, samples in by_question_type.items():
        if len(samples) == 0:
            continue
            
        # Shuffle samples within this question type
        samples_copy = samples.copy()
        random.shuffle(samples_copy)
        
        n = len(samples_copy)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        # Split this question type
        type_train = samples_copy[:train_end]
        type_val = samples_copy[train_end:val_end]
        type_test = samples_copy[val_end:]
        
        # Add to overall splits
        train_split.extend(type_train)
        val_split.extend(type_val)
        test_split.extend(type_test)
        
        print(f"{question_type}: {len(type_train)} train + {len(type_val)} val + {len(type_test)} test")
    
    # Final shuffle to mix question types within each split
    random.shuffle(train_split)
    random.shuffle(val_split)
    random.shuffle(test_split)
    
    print(f"Stratified split result: {len(train_split)} train + {len(val_split)} val + {len(test_split)} test")
    
    # Verify balance is maintained
    print("Final distribution verification:")
    for split_name, split_data in [("Train", train_split), ("Val", val_split), ("Test", test_split)]:
        if len(split_data) > 0:
            split_distribution = defaultdict(int)
            for sample in split_data:
                q_type = get_question_type(sample['question'])
                split_distribution[q_type] += 1
            
            print(f"{split_name}: {dict(split_distribution)}")
    
    return train_split, val_split, test_split

def check_answer_indices(qa_samples, num_samples=10):
    """
    Simple function to check if answer indices are correct
    Just add this anywhere in your script and call it
    """
    print("CHECKING ANSWER INDICES")
    print("="*50)
    
    # Check first N samples or all if fewer
    samples_to_check = qa_samples[:num_samples] if len(qa_samples) > num_samples else qa_samples
    
    correct_count = 0
    total_checked = 0
    
    for i, sample in enumerate(samples_to_check):
        print(f"\n--- Sample {i+1} ---")
        
        context = sample.get('context', '')
        answer_text = sample.get('answer_text', '')
        answer_start = sample.get('answer_start', -1)
        sample_id = sample.get('id', f'sample_{i}')
        
        print(f"ID: {sample_id}")
        print(f"Question: {sample['question']}")
        print(f"Expected Answer: '{answer_text}'")
        print(f"Answer Start Index: {answer_start}")
        
        # Skip empty answers
        if not answer_text or answer_text.strip() == '':
            print("Empty answer (OK)")
            correct_count += 1
            total_checked += 1
            continue
        
        # Check if index is valid
        if answer_start < 0 or answer_start >= len(context):
            print(f"INVALID INDEX: {answer_start} (context length: {len(context)})")
            total_checked += 1
            continue
        
        # Extract text at the index
        answer_end = answer_start + len(answer_text)
        extracted_text = context[answer_start:answer_end]
        
        print(f"Text at Index {answer_start}-{answer_end}: '{extracted_text}'")
        
        # Show context around the answer
        start_preview = max(0, answer_start - 20)
        end_preview = min(len(context), answer_end + 20)
        context_preview = context[start_preview:end_preview]
        highlighted = context_preview.replace(extracted_text, f">>>{extracted_text}<<<")
        print(f"Context: ...{highlighted}...")
        
        # Check if they match
        if extracted_text == answer_text:
            print("CORRECT INDEX")
            correct_count += 1
        else:
            print("WRONG INDEX")
            # Try to find where it should be
            correct_index = context.find(answer_text)
            if correct_index != -1:
                print(f"Should be at index: {correct_index}")
            else:
                print(f"Answer text not found in context!")
        
        total_checked += 1
        print("-" * 30)
    
    print(f"\nSUMMARY:")
    print(f"Checked: {total_checked} samples")
    print(f"Correct: {correct_count}")
    print(f"Wrong: {total_checked - correct_count}")
    print(f"Accuracy: {correct_count/total_checked*100:.1f}%")
    
    return correct_count, total_checked

import sys
from datetime import datetime
import contextlib

# Class for capturing output
class OutputLogger:
    """Captures both console output and saves to file"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)  # Show on console
        self.log.write(message)       # Also save to file
        self.log.flush()              # Ensure immediate writing
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()

def reset_reused_flags(review_file="reviewed_samples.json"):
    """Reset all reused flags to False and remove last_used_iteration for fresh training start"""
    if not os.path.exists(review_file):
        print("No review file found to reset")
        return
    
    try:
        with open(review_file, 'r') as f:
            reviews = json.load(f)
        
        reset_count = 0
        removed_last_used_count = 0
        
        for review_key, review_data in reviews.items():
            # Reset reused flag if it was True
            if review_data.get('reused', False) == True:
                reviews[review_key]['reused'] = False
                reset_count += 1
            
            # Remove last_used_iteration field if it exists (from previous runs)
            if 'last_used_iteration' in review_data:
                del reviews[review_key]['last_used_iteration']
                removed_last_used_count += 1
        
        with open(review_file, 'w') as f:
            json.dump(reviews, f, indent=2)
        
        print(f"Reset {reset_count} samples to reused=False")
        print(f"Removed {removed_last_used_count} last_used_iteration fields from previous runs")
        
    except Exception as e:
        print(f"Error resetting reused flags: {e}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main self-training pipeline using real annotated data for initial training"""
    
    # Create output logger with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"training_output_{timestamp}.log"
    
    # Redirect stdout to capture all output
    output_logger = OutputLogger(output_filename)
    original_stdout = sys.stdout
    sys.stdout = output_logger
    
    try:
        print("SafetyBERT Self-Training QA System - Real Annotated Data Version")
        print("=" * 70)
        print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output being saved to: {output_filename}")
        print("=" * 70)

        # NEW: Reset all reused flags for fresh training start
        print("\nResetting all reused flags for fresh training...")
        reset_reused_flags()
        
        # Phase 1: Load real annotated data and train base model
        print("\nPHASE 1: Real Data Loading & Base Model Training")
        print("-" * 50)
        
        # Load your annotated data
        annotated_data = load_annotated_data("annotated-data.csv")
        
        if not annotated_data:
            print("Failed to load annotated data")
            return
        
        # Convert to SQuAD format with your questions
        squad_data = convert_annotated_to_squad_format(annotated_data)
        
        if not squad_data:
            print("Failed to convert annotated data to QA format")
            return
        
        # Split annotated data for training
        train_data, val_data, test_data = create_train_val_test_split(squad_data, 0.7, 0.15, 0.15)
        
        # Setup model
        model, tokenizer = setup_model_and_tokenizer("/home/exouser/QA-traninig-self-learning-MSHA/safetyBERT/")
        if model is None or tokenizer is None:
            return
        
        # Preprocess data
        train_dataset = preprocess_data(train_data, tokenizer)
        val_dataset = preprocess_data(val_data, tokenizer)
        
        # Train base model on real annotated data
        trainer = create_trainer(model, tokenizer, train_dataset, val_dataset)
        
        print("\nTraining base model on real annotated data...")
        train_result = trainer.train()
        
        # Save base model
        model.save_pretrained("./base_safety_qa_model_real_data")
        tokenizer.save_pretrained("./base_safety_qa_model_real_data")
        print("Base model saved to ./base_safety_qa_model_real_data")
        
        # Evaluate base model on real test set
        base_score = evaluate_model_qa(model, tokenizer, test_data, "Base Model - Real Annotated Test Data")
        
        # Phase 2: Load prediction pool (external accident narratives for self-training)
        print("\nPHASE 2: Loading Prediction Pool for Self-Training")
        print("-" * 50)
        
        # Load external narratives for self-training (different from training data)
        all_accident_narratives = load_accident_narratives(file_path="data-mill-without-annotated-samples-and-fatality-occupational-illness.csv")
        if not all_accident_narratives:
            print("No external narratives available for self-training")
            return
        
        # Generate synthetic QA pairs for external narratives using question templates
        all_synthetic_qa_pairs = generate_questions_for_narratives(all_accident_narratives)
        print(f"Generated {len(all_synthetic_qa_pairs)} total QA pairs from {len(all_accident_narratives)} external narratives")
        
        # Phase 3: Self-training loop
        print("\nPHASE 3: Self-Training Loop")
        print("-" * 40)
        
        # Generate predictions for ALL QA pairs ONCE
        print("Generating predictions for all QA pairs (one-time processing)...")
        all_predictions = predict_with_confidence(model, tokenizer, all_synthetic_qa_pairs)   
        
        # Sort by confidence (lowest first) for efficient sampling
        all_predictions.sort(key=lambda x: x['confidence'], reverse=False)
        print(f"Sorted {len(all_predictions)} predictions by confidence")
        print(f"Confidence range: {all_predictions[-1]['confidence']:.3f} - {all_predictions[0]['confidence']:.3f}")
        
        # Initialize datasets - START WITH ORIGINAL ANNOTATED DATA AS FOUNDATION
        current_training_data = train_data.copy()      
        current_validation_data = val_data.copy()      
        current_test_data = test_data.copy()       
        
        best_score = base_score
        max_iterations = 20
        used_prediction_indices = set()  # Track which predictions we've already used
        
        performance_history = [{'iteration': 0,
                                'score': base_score,
                                'training_size': len(current_training_data),
                                'validation_size': len(current_validation_data),    
                                'test_size': len(current_test_data),
                                'approved_samples': 0 }]
        
        # User-configurable parameters
        SAMPLES_PER_ITERATION = 200  
        
        print(f"Configuration:")
        print(f"Samples per iteration: {SAMPLES_PER_ITERATION}")
        print(f"Max iterations: {max_iterations}")

        early_stopped = False
        best_score_iteration = 0
        
        for iteration in range(1, max_iterations + 1):
            print(f"\nITERATION {iteration}")
            print("=" * 30)
            
            if iteration == 1:
                print('Using base model predictions for iteration 1')
                current_predictions = all_predictions
                # For iteration 1, predictions don't have original_index, so add them
                for i, pred in enumerate(current_predictions):
                    pred['original_index'] = i
            else:
                # Generate fresh predictions with updated model
                print(f'Generating fresh predictions with iteration {iteration} model...')

                unused_qa_pairs = []
                unused_original_indices = []

                for i, qa in enumerate(all_synthetic_qa_pairs):
                    if i not in used_prediction_indices:
                        unused_qa_pairs.append(qa)
                        unused_original_indices.append(i)

                if len(unused_qa_pairs) == 0:
                    print("All QA pairs used - stopping training")
                    break
                            
                # Generate predictions with CURRENT (improved) model
                fresh_predictions  = predict_with_confidence(model, tokenizer, unused_qa_pairs)
                current_predictions = []
                for j, pred in enumerate(fresh_predictions):
                    pred['original_index'] = unused_original_indices[j]  # Add original index
                    current_predictions.append(pred)
                                
                current_predictions.sort(key=lambda x: x['confidence'], reverse=False)
                
                print(f'Generated {len(current_predictions)} fresh predictions')
                if current_predictions:
                    confidences = [pred.get('confidence', -1) for pred in current_predictions]
                    print(f'Confidence range: {min(confidences):.3f} - {max(confidences):.3f}')
                    print(f'Average confidence: {sum(confidences)/len(confidences):.3f}')
            
            # Select low-confidence samples for this iteration
            hybrid_samples = select_hybrid_samples(
                current_predictions, 
                used_prediction_indices,
                max_samples=SAMPLES_PER_ITERATION
            )
                  
            if len(hybrid_samples) == 0:
                print("No samples found for hybrid selection. Stopping self-training.")
                break

            print(f"HUMAN REVIEW: {len(hybrid_samples)} selected samples (hybrid strategy)")
            
            # HUMAN REVIEW
            reviewed_samples, all_processed_indices = interactive_review(hybrid_samples, iteration, all_synthetic_qa_pairs)

            # Add ALL processed indices (both reused and newly reviewed) to used_prediction_indices
            for idx in all_processed_indices:
                used_prediction_indices.add(idx)

            print(f"Added {len(all_processed_indices)} processed samples to used_prediction_indices")
            print(f"Total used samples: {len(used_prediction_indices)}/{len(all_synthetic_qa_pairs)}")
            
            approved_samples = convert_reviewed_samples_to_training_format(reviewed_samples)

            print(f"\nChecking approved samples BEFORE conversion...")
            samples_with_answers = [s for s in approved_samples if s.get('answer_text', '').strip() != '']
            print(f"Found {len(samples_with_answers)} samples with actual answers out of {len(approved_samples)} total")
            
            if len(approved_samples) == 0:
                print("No samples approved by human review. Stopping self-training.")
                break

            print(f"Human approved: {len(approved_samples)}/{len(hybrid_samples)} samples")
            print(f'Total used QA pairs: {len(used_prediction_indices)}/{len(all_synthetic_qa_pairs)}')

            # **SPLIT APPROVED SAMPLES 70:15:15**
            approved_train, approved_val, approved_test = split_approved_samples(approved_samples)
            
            if len(approved_train) == 0 and len(approved_val) == 0 and len(approved_test) == 0:
                print("No samples to add after splitting - continuing...")
                continue
            
            # Convert each split to training format
            pseudo_train_data = convert_to_training_format(approved_train)
            pseudo_val_data = convert_to_training_format(approved_val)
            pseudo_test_data = convert_to_training_format(approved_test)
            
            # Add to respective datasets (FOUNDATION + NEW SAMPLES)
            current_training_data.extend(pseudo_train_data)
            current_validation_data.extend(pseudo_val_data)
            current_test_data.extend(pseudo_test_data)

            # Update reused flags for all approved samples (so they won't be selected again)
            print(f"Updating reused status for all approved samples...")
            try:
                # Load existing reviews
                if os.path.exists("reviewed_samples.json"):
                    with open("reviewed_samples.json", 'r') as f:
                        existing_reviews = json.load(f)
                else:
                    existing_reviews = {}
                
                # Mark ALL approved samples as reused (so they won't be selected again)
                marked_count = 0
                for sample in approved_samples:
                    sample_key = create_sample_key(sample['context'], sample['question'])
                    
                    # Find and update the matching review entry
                    for review_key, review_data in existing_reviews.items():
                        existing_sample_key = create_sample_key(review_data['narrative'], review_data['question'])
                        if existing_sample_key == sample_key:
                            # Update reused flag for approved samples
                            decision = review_data.get('your_decision', '')
                            if decision in ['accepted', 'corrected']:
                                existing_reviews[review_key]['reused'] = True
                                existing_reviews[review_key]['last_used_iteration'] = iteration
                                marked_count += 1
                            break
                
                # Save updated reviews
                with open("reviewed_samples.json", 'w') as f:
                    json.dump(existing_reviews, f, indent=2)
                
                print(f"Marked {marked_count} approved samples as reused=True")
                
            except Exception as e:
                print(f"Error marking samples as reused: {e}")
            
            print(f"Updated dataset sizes:")
            print(f"Training: {len(current_training_data)} (+{len(pseudo_train_data)} new)")
            print(f"Validation: {len(current_validation_data)} (+{len(pseudo_val_data)} new)")
            print(f"Test: {len(current_test_data)} (+{len(pseudo_test_data)} new)")

            # Retrain model with expanded datasets
            train_dataset_expanded = preprocess_data(current_training_data, tokenizer)
            val_dataset_expanded = preprocess_data(current_validation_data, tokenizer)

            print(f"Loading fresh BERT model for iteration {iteration}")
            model, tokenizer = setup_model_and_tokenizer("/home/exouser/QA-traninig-self-learning-MSHA/safetyBERT/")
            
            trainer_expanded = create_trainer(model, tokenizer, train_dataset_expanded, val_dataset_expanded)
            print(f"Training from scratch on {len(current_training_data)} total samples")
            train_result = trainer_expanded.train()
            
            # Evaluate improved model on current validation set
            current_score = evaluate_model_qa(model, tokenizer, current_validation_data, f"Iteration {iteration} - Current Validation", print_wrong_predictions=False)
            
            # Track performance
            performance_history.append({
                'iteration': iteration, 
                'score': current_score, 
                'training_size': len(current_training_data),
                'validation_size': len(current_validation_data),
                'test_size': len(current_test_data),
                'approved_samples': len(approved_samples)
            })
            
            print(f"Iteration {iteration} Score: {current_score:.1f}% (vs Base: {base_score:.1f}%)")

            # Early stopping check
            SL_EARLY_STOPPING_PATIENCE = 3  # Allow 3 declining iterations

            if current_score > best_score:
                best_score = current_score
                best_score_iteration = iteration  # Update when best was achieved
                print(f"New best score: {best_score:.1f}% at iteration {iteration}")
                
                # Save improved model
                model.save_pretrained(f"./self_trained_model_iter_{iteration}")
                tokenizer.save_pretrained(f"./self_trained_model_iter_{iteration}")
                print(f"Best model saved to ./best_model_iter_{iteration}")
                
            iterations_since_best = iteration - best_score_iteration
            
            if iterations_since_best >= SL_EARLY_STOPPING_PATIENCE:
                print(f'No improvement for {SL_EARLY_STOPPING_PATIENCE} iterations since best score!')
                print(f'Best score: {best_score:.1f}% (iteration {best_score_iteration})')
                print(f'Current score: {current_score:.1f}% (iteration {iteration})')

                # Save the final model before stopping
                print(f"Saving final iteration model before early stopping...")
                model.save_pretrained(f"./final_model_iter_{iteration}")
                tokenizer.save_pretrained(f"./final_model_iter_{iteration}")
                print(f"Final model saved to ./final_model_iter_{iteration}")

                early_stopped = True

                print(f'Stopping early...')
                break

            print(f"Iterations since best: {iterations_since_best}/{SL_EARLY_STOPPING_PATIENCE}")
                
            if current_score < best_score - 70:

                early_stopped = True
            
                print("Significant performance degradation. Stopping self-training.")
                break
            else:
                print("Continuing training...")

        if not early_stopped:
            print(f"Maximum iterations ({max_iterations}) reached.")
            print(f"Saving final iteration model...")
            model.save_pretrained(f"./final_model_iter_{iteration}")
            tokenizer.save_pretrained(f"./final_model_iter_{iteration}")
            print(f"Final model saved to ./final_model_iter_{iteration}")

        # Final evaluation summary
        print("\nFINAL RESULTS")
        print("=" * 40)
        
        print("Performance History:")
        for hist in performance_history:
            approved = hist.get('approved_samples', 'N/A')
            print(f"Iteration {hist['iteration']}: {hist['score']:.1f}% "
                  f"(Train: {hist['training_size']}, Val: {hist.get('validation_size', 'N/A')}, "
                  f"Test: {hist.get('test_size', 'N/A')}, Approved: {approved})")
        
        improvement = best_score - base_score
        print(f"\nFinal Improvement: {improvement:+.1f}% (Base: {base_score:.1f}% ? Best: {best_score:.1f}%)")
        
        if improvement > 0:
            print("Self-training successful!")
        else:
            print("Self-training did not improve performance")
        
        # Final evaluation on accumulated test data
        print("\nFINAL EVALUATION ON ACCUMULATED TEST DATA")
        print("=" * 50)

        # Load the best saved model for final evaluation
        if best_score > base_score:  # If we found any improvements
            import glob
            model_dirs = glob.glob("./self_trained_model_iter_*")
            if model_dirs:
                # Find the model with the highest iteration number (most recent best)
                latest_best_model = max(model_dirs, key=lambda x: int(x.split('_')[-1]))
                print(f"Loading best model from {latest_best_model} for final evaluation...")
                model = BertForQuestionAnswering.from_pretrained(latest_best_model)
                tokenizer = AutoTokenizer.from_pretrained(latest_best_model)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.to(device)
                print(f"Best model loaded (best score: {best_score:.1f}%)")
            else:
                print("No saved iteration models found, using current model")
        else:
            print("No improvements found, using base model for final evaluation")
     
        print(f"Evaluating on {len(current_test_data)} accumulated test samples...")
        final_test_score = evaluate_model_qa(model, tokenizer, current_test_data, "Final Accumulated Test Set", print_wrong_predictions=True)
        
        print(f"Final Test Score: {final_test_score:.1f}%")

        print(f"\nSAVING DETAILED TEST RESULTS")
        print("=" * 40)
        
        try:
            print(f"Generating detailed predictions for {len(current_test_data)} test samples...")
            
            detailed_test_results = []
            model.eval()
            device = next(model.parameters()).device
            
            for i, test_sample in enumerate(current_test_data):
                # Get model prediction
                inputs = tokenizer(
                    test_sample["question"],
                    test_sample["context"], 
                    return_tensors="pt",
                    max_length=128,
                    truncation=True,
                    padding=True
                )
                
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                
                start_idx = torch.argmax(outputs.start_logits[0]).item()
                end_idx = torch.argmax(outputs.end_logits[0]).item()
                
                if start_idx <= end_idx and start_idx < len(inputs["input_ids"][0]) and end_idx < len(inputs["input_ids"][0]):
                    predicted_answer = tokenizer.decode(
                        inputs["input_ids"][0][start_idx:end_idx+1],
                        skip_special_tokens=True
                    ).strip()
                else:
                    predicted_answer = ""
                
                # Get actual answer(s)
                if test_sample["answers"]["text"]:
                    actual_answers = test_sample["answers"]["text"] if isinstance(test_sample["answers"]["text"], list) else [test_sample["answers"]["text"]]
                    actual_answer = actual_answers[0]  # Use first answer as primary
                    all_actual_answers = actual_answers
                else:
                    actual_answer = ""
                    all_actual_answers = []
                
                # Determine question type
                question_type = get_question_type(test_sample['question'])
                
                # Calculate if prediction is correct (exact match)
                is_correct = any(
                    predicted_answer.lower().strip() == ans.lower().strip() 
                    for ans in all_actual_answers
                ) if all_actual_answers else (predicted_answer.strip() == "")
                
                # Create detailed result entry
                result_entry = {
                    "sample_id": test_sample.get("id", f"test_sample_{i}"),
                    "question_type": question_type,
                    "narrative": test_sample["context"],
                    "question": test_sample["question"],
                    "actual_answer": actual_answer,
                    "all_actual_answers": all_actual_answers,
                    "predicted_answer": predicted_answer,
                    "is_correct": is_correct,
                    "is_impossible": test_sample.get("is_impossible", False),
                    "confidence_scores": {
                        "start_logit_max": float(torch.max(outputs.start_logits[0]).item()),
                        "end_logit_max": float(torch.max(outputs.end_logits[0]).item())
                    }
                }
                
                detailed_test_results.append(result_entry)
                
                # Progress indicator
                if (i + 1) % 50 == 0:
                    print(f"Processed {i + 1}/{len(current_test_data)} samples...")
            
            # Save detailed results
            with open("detailed_test_results.json", "w") as f:
                json.dump(detailed_test_results, f, indent=2)
            print("Saved: detailed_test_results.json")
            
            # Generate summary statistics
            total_samples = len(detailed_test_results)
            correct_predictions = sum(1 for r in detailed_test_results if r["is_correct"])
            accuracy = (correct_predictions / total_samples * 100) if total_samples > 0 else 0
            
            # By question type
            by_question_type = {}
            for result in detailed_test_results:
                q_type = result["question_type"]
                if q_type not in by_question_type:
                    by_question_type[q_type] = {"total": 0, "correct": 0}
                by_question_type[q_type]["total"] += 1
                if result["is_correct"]:
                    by_question_type[q_type]["correct"] += 1
            
            # Save summary
            summary = {
                "total_test_samples": total_samples,
                "correct_predictions": correct_predictions,
                "overall_accuracy": accuracy,
                "final_test_score": final_test_score,
                "by_question_type": {
                    q_type: {
                        "total": stats["total"],
                        "correct": stats["correct"],
                        "accuracy": (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
                    }
                    for q_type, stats in by_question_type.items()
                }
            }
            
            with open("test_results_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            print("Saved: test_results_summary.json")
            
            print(f"\nTest Results Summary:")
            print(f"Total samples: {total_samples}")
            print(f"Correct predictions: {correct_predictions}")
            print(f"Overall accuracy: {accuracy:.1f}%")
            print(f"F1 score: {final_test_score:.1f}%")
            
            print(f"\nBy Question Type:")
            for q_type, stats in by_question_type.items():
                acc = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
                print(f"{q_type}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
            
        except Exception as e:
            print(f"Error generating detailed test results: {e}")
        
        # Save final accumulated datasets in SQuAD v2.0 format
        print(f"\nSAVING FINAL ACCUMULATED DATASETS")
        print("=" * 40)
        
        try:
            # Save training data
            print(f"Saving training data: {len(current_training_data)} samples...")
            with open("final_train_data.json", "w") as f:
                json.dump(current_training_data, f, indent=2)
            print("Saved: final_train_data.json")
            
            # Save validation data
            print(f"Saving validation data: {len(current_validation_data)} samples...")
            with open("final_val_data.json", "w") as f:
                json.dump(current_validation_data, f, indent=2)
            print("Saved: final_val_data.json")
            
            # Save test data
            print(f"Saving test data: {len(current_test_data)} samples...")
            with open("final_test_data.json", "w") as f:
                json.dump(current_test_data, f, indent=2)
            print("Saved: final_test_data.json")
            
            # Save summary statistics
            dataset_summary = {
                "training_started": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "total_iterations": len(performance_history) - 1,  # -1 because includes base (iteration 0)
                "final_training_size": len(current_training_data),
                "final_validation_size": len(current_validation_data),
                "final_test_size": len(current_test_data),
                "total_samples": len(current_training_data) + len(current_validation_data) + len(current_test_data),
                "base_score": base_score,
                "best_score": best_score,
                "final_test_score": final_test_score,
                "final_improvement": best_score - base_score,
                "used_predictions": len(used_prediction_indices),
                "total_available_predictions": len(all_synthetic_qa_pairs),
                "utilization_rate": len(used_prediction_indices) / len(all_synthetic_qa_pairs) * 100,
                "performance_history": performance_history
            }
            
            with open("final_dataset_summary.json", "w") as f:
                json.dump(dataset_summary, f, indent=2)
            print("? Saved: final_dataset_summary.json")
            
            print(f"\nFinal Dataset Summary:")
            print(f"Training: {len(current_training_data)} samples")
            print(f"Validation: {len(current_validation_data)} samples") 
            print(f"Test: {len(current_test_data)} samples")
            print(f"Total: {len(current_training_data) + len(current_validation_data) + len(current_test_data)} samples")
            print(f"Used {len(used_prediction_indices)}/{len(all_synthetic_qa_pairs)} available predictions ({len(used_prediction_indices)/len(all_synthetic_qa_pairs)*100:.1f}%)")
            
        except Exception as e:
            print(f"Error saving final datasets: {e}")

        # Save models and results
        print(f"\nBest model saved to ./self_trained_model_iter_*")
        
        # Save training history
        with open("self_training_history.json", "w") as f:
            json.dump(performance_history, f, indent=2)
        
        print(f"\nSelf-training pipeline with real annotated data completed!")
        print(f"Complete training log saved to: {output_filename}")
        print(f"Training ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Always restore stdout and close log file
        sys.stdout = original_stdout
        output_logger.close()
        print(f"\nTraining output saved to: {output_filename}")

if __name__ == "__main__":
    main()
