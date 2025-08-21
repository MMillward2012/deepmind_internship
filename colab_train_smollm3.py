#!/usr/bin/env python3
"""
SmolLM3 Financial Sentiment Training Script for Google Colab
============================================================

This script trains SmolLM3-3B on financial sentiment data in Google Colab.
Optimized for Colab's T4 GPU with 15GB VRAM.

Usage in Colab:
1. Upload this script and your data
2. Run: !python colab_train_smollm3.py
3. Download the trained model

Author: Training Pipeline
"""

import os
import sys
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
CONFIG = {
    "model_id": "HuggingFaceTB/SmolLM3-3B",
    "model_name": "smollm3-financial-sentiment",
    "num_labels": 3,
    "max_length": 512,
    "batch_size": 2,  # Conservative for 15GB GPU
    "gradient_accumulation_steps": 8,  # Effective batch size = 16
    "learning_rate": 1e-5,  # Lower LR for large model
    "num_epochs": 2,  # Quick training
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "save_steps": 500,
    "logging_steps": 50,
    "eval_steps": 500,
    "fp16": True,  # Enable mixed precision
    "dataloader_num_workers": 2,
}

def setup_colab_environment():
    """Setup the Colab environment with required packages."""
    print("🔧 Setting up Colab environment...")
    
    # Install required packages
    os.system("pip install transformers datasets accelerate -q")
    os.system("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q")
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU detected: {gpu_name}")
        print(f"💾 GPU memory: {gpu_memory:.1f} GB")
        
        if gpu_memory < 12:
            print("⚠️ Warning: GPU memory might be insufficient for SmolLM3-3B")
            CONFIG["batch_size"] = 1
            CONFIG["gradient_accumulation_steps"] = 16
            print(f"🔧 Adjusted batch size to {CONFIG['batch_size']}")
    else:
        print("❌ No GPU detected - training will be very slow!")
        
    return torch.cuda.is_available()

def load_financial_data():
    """Load and preprocess financial sentiment data."""
    print("📊 Loading financial sentiment data...")
    
    # Try to load from multiple possible locations
    data_paths = [
        "data/FinancialPhraseBank/all-data.csv",
        "/content/data/FinancialPhraseBank/all-data.csv",
        "/content/all-data.csv",
        "all-data.csv"
    ]
    
    data_df = None
    for path in data_paths:
        if Path(path).exists():
            print(f"📂 Found data at: {path}")
            data_df = pd.read_csv(path, 
                                names=['text', 'label'], 
                                encoding='utf-8', 
                                on_bad_lines='skip')
            break
    
    if data_df is None:
        print("❌ No data file found! Please upload your data file.")
        print("Expected locations:", data_paths)
        return None, None, None
    
    # Clean and preprocess
    data_df = data_df.dropna()
    data_df['text'] = data_df['text'].astype(str)
    data_df['label'] = data_df['label'].astype(str)
    
    # Map sentiment labels to integers
    label_mapping = {
        'positive': 2,
        'neutral': 1, 
        'negative': 0
    }
    
    # Try different label formats
    if not data_df['label'].isin(['positive', 'neutral', 'negative']).all():
        # Check if labels are already numeric or need other mapping
        unique_labels = data_df['label'].unique()
        print(f"🏷️ Found labels: {unique_labels}")
        
        # Auto-create mapping for numeric labels
        if all(str(label).replace('.', '').replace('-', '').isdigit() for label in unique_labels):
            sorted_labels = sorted(unique_labels, key=lambda x: float(x))
            label_mapping = {str(label): i for i, label in enumerate(sorted_labels)}
            print(f"🔄 Auto-mapped labels: {label_mapping}")
    
    data_df['label_id'] = data_df['label'].map(label_mapping)
    data_df = data_df.dropna(subset=['label_id'])
    data_df['label_id'] = data_df['label_id'].astype(int)
    
    print(f"✅ Loaded {len(data_df)} samples")
    print(f"📊 Label distribution: {dict(data_df['label_id'].value_counts())}")
    
    # Train/val split
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        data_df, test_size=0.2, random_state=42, stratify=data_df['label_id']
    )
    
    print(f"🔄 Split: {len(train_df)} train, {len(val_df)} validation")
    
    return train_df, val_df, label_mapping

def create_dataset(df, tokenizer, max_length):
    """Create a HuggingFace dataset from pandas DataFrame."""
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors=None
        )
    
    # Prepare dataset
    dataset = Dataset.from_pandas(df[['text', 'label_id']].rename(columns={'label_id': 'labels'}))
    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text'],
        num_proc=1
    )
    
    return dataset

def train_smollm3(train_df, val_df, label_mapping):
    """Train SmolLM3 on financial sentiment data."""
    print("🚀 Starting SmolLM3 training...")
    
    # Load tokenizer and model
    print("📥 Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_id"])
    
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("🔧 Set pad_token to eos_token")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["model_id"],
        num_labels=CONFIG["num_labels"],
        id2label={0: 'negative', 1: 'neutral', 2: 'positive'},
        label2id={'negative': 0, 'neutral': 1, 'positive': 2}
    )
    
    print(f"✅ Model loaded: {model.config.__class__.__name__}")
    print(f"📊 Parameters: {model.num_parameters():,}")
    
    # Create datasets
    print("🔄 Creating datasets...")
    train_dataset = create_dataset(train_df, tokenizer, CONFIG["max_length"])
    val_dataset = create_dataset(val_df, tokenizer, CONFIG["max_length"])
    
    # Training arguments
    output_dir = f"/content/{CONFIG['model_name']}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=CONFIG["num_epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
        warmup_steps=CONFIG["warmup_steps"],
        logging_steps=CONFIG["logging_steps"],
        save_steps=CONFIG["save_steps"],
        eval_steps=CONFIG["eval_steps"],
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,
        fp16=CONFIG["fp16"],
        dataloader_num_workers=CONFIG["dataloader_num_workers"],
        save_total_limit=2,
        logging_dir=f"{output_dir}/logs",
        push_to_hub=False,
        remove_unused_columns=True,
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors='pt'
    )
    
    # Metrics
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(labels, predictions)
        return {'accuracy': accuracy}
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("🏋️ Starting training...")
    print(f"⏱️ Estimated time: ~{CONFIG['num_epochs'] * len(train_dataset) // (CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']) * 2 // 60} minutes")
    
    train_result = trainer.train()
    
    # Evaluate
    print("📊 Running final evaluation...")
    eval_result = trainer.evaluate()
    
    # Save model
    print("💾 Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save results
    results = {
        'training_timestamp': datetime.now().isoformat(),
        'model_id': CONFIG["model_id"],
        'model_name': CONFIG["model_name"],
        'config': CONFIG,
        'label_mapping': label_mapping,
        'train_loss': train_result.training_loss,
        'eval_loss': eval_result['eval_loss'],
        'eval_accuracy': eval_result['eval_accuracy'],
        'train_runtime': train_result.metrics.get('train_runtime', 0),
        'samples_per_second': train_result.metrics.get('train_samples_per_second', 0)
    }
    
    with open(f"{output_dir}/training_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("🎉 TRAINING COMPLETED!")
    print(f"{'='*60}")
    print(f"📊 Final Results:")
    print(f"   🎯 Validation accuracy: {eval_result['eval_accuracy']:.4f}")
    print(f"   📉 Validation loss: {eval_result['eval_loss']:.4f}")
    print(f"   ⏱️ Training time: {train_result.metrics.get('train_runtime', 0):.1f}s")
    print(f"   📁 Model saved to: {output_dir}")
    print(f"\n💾 To download your model:")
    print(f"   1. Zip the model directory: !zip -r {CONFIG['model_name']}.zip {output_dir}")
    print(f"   2. Download from Colab Files panel")
    
    return output_dir, results

def main():
    """Main training pipeline."""
    print("🚀 SmolLM3 Financial Sentiment Training")
    print("="*50)
    
    # Setup environment
    has_gpu = setup_colab_environment()
    if not has_gpu:
        response = input("⚠️ No GPU detected. Continue with CPU training? (very slow) [y/N]: ")
        if response.lower() != 'y':
            print("❌ Training cancelled")
            return
    
    # Load data
    train_df, val_df, label_mapping = load_financial_data()
    if train_df is None:
        print("❌ Failed to load data")
        return
    
    # Train model
    model_dir, results = train_smollm3(train_df, val_df, label_mapping)
    
    # Create download package
    print("\n📦 Creating download package...")
    os.system(f"zip -r {CONFIG['model_name']}.zip {model_dir}")
    print(f"✅ Created {CONFIG['model_name']}.zip")
    
    print("\n🎊 Training complete! Your SmolLM3 model is ready.")

if __name__ == "__main__":
    main()
