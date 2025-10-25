#!/usr/bin/env python3
"""
RoBERTa Multiclass Classifier for Homelessness Content Analysis

This script implements a RoBERTa-based multiclass classifier that excels at classifying
homelessness-related content across all 16 categories in a single model.

RoBERTa typically performs 2-5% better than BERT on classification tasks due to:
- More robust pre-training with dynamic masking
- Better optimization of the training procedure
- Improved handling of longer sequences

Usage:
    python scripts/roberta_multiclass_classifier.py --source reddit --mode train
    python scripts/roberta_multiclass_classifier.py --source reddit --mode evaluate
    python scripts/roberta_multiclass_classifier.py --source reddit --mode predict --input data.csv
"""

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split
import json
import os
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# Define all 16 categories
ALL_CATEGORIES = [
    # Comment Types (6)
    'ask a genuine question',
    'ask a rhetorical question',
    'provide a fact or claim',
    'provide an observation',
    'express their opinion',
    'express others opinions',
    # Critique Categories (3)
    'money aid allocation',
    'government critique',
    'societal critique',
    # Response Categories (1)
    'solutions/interventions',
    # Perception Types (5)
    'personal interaction',
    'media portrayal',
    'not in my backyard',
    'harmful generalization',
    'deserving/undeserving',
    # Racist Classification (1)
    'racist'
]

class HomelessnessDataset(Dataset):
    """Dataset for homelessness content classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(label)
        }

def load_and_preprocess_data(source):
    """Load and preprocess data for a specific source
    
    Soft labels are weighted based on annotator agreement:
    - 0: 0 annotators agreed
    - 0.33: 1 annotator agreed  
    - 0.67: 2 annotators agreed
    - 1: 3 annotators agreed
    
    We threshold at >= 0.5 to treat 2+ annotator agreement as positive.
    """
    print(f"Loading data for {source}...")
    
    # Load soft labels
    soft_labels_path = f'output/annotation/soft_labels/{source}_soft_labels.csv'
    if not os.path.exists(soft_labels_path):
        raise FileNotFoundError(f"Soft labels file not found: {soft_labels_path}")
    
    soft_labels_df = pd.read_csv(soft_labels_path)
    print(f"Loaded {len(soft_labels_df)} soft label samples")
    
    # Load text data from gold standard files
    gold_standard_map = {
        'reddit': 'gold_standard/sampled_reddit_comments.csv',
        'x': 'gold_standard/sampled_twitter_posts.csv',
        'news': 'gold_standard/sampled_lexisnexis_news.csv',
        'meeting_minutes': 'gold_standard/sampled_meeting_minutes.csv'
    }
    
    text_file_path = gold_standard_map.get(source)
    if not os.path.exists(text_file_path):
        raise FileNotFoundError(f"Text data file not found: {text_file_path}")
    
    text_df = pd.read_csv(text_file_path)
    print(f"Loaded {len(text_df)} text samples")
    
    # Get text column based on source
    text_col_map = {
        'reddit': 'Comment',
        'x': 'Deidentified_text', 
        'news': 'Deidentified_paragraph_text',
        'meeting_minutes': 'Deidentified_paragraph'
    }
    
    text_col = text_col_map.get(source)
    if text_col not in text_df.columns:
        raise ValueError(f"Text column '{text_col}' not found in text data")
    
    # Ensure we have the same number of samples
    if len(soft_labels_df) != len(text_df):
        print(f"Warning: Soft labels ({len(soft_labels_df)}) and text data ({len(text_df)}) have different lengths")
        # Use the minimum length
        min_len = min(len(soft_labels_df), len(text_df))
        soft_labels_df = soft_labels_df.iloc[:min_len]
        text_df = text_df.iloc[:min_len]
    
    texts = text_df[text_col].fillna('').astype(str).tolist()
    
    # Create labels matrix for all 16 categories
    labels = np.zeros((len(soft_labels_df), len(ALL_CATEGORIES)))
    
    # Map soft label columns to our categories
    for i, category in enumerate(ALL_CATEGORIES):
        if category in soft_labels_df.columns:
            # Convert soft labels to binary (threshold at 0.5)
            # Soft labels are: 0 (0 annotators), 0.33 (1 annotator), 0.67 (2 annotators), 1 (3 annotators)
            # Treat >= 0.5 as positive (2 or 3 annotators agreed)
            labels[:, i] = (soft_labels_df[category] >= 0.5).astype(int)
        else:
            print(f"Warning: Category '{category}' not found in data, using zeros")
    
    print(f"Label distribution (threshold >= 0.5 for 2+ annotators):")
    for i, category in enumerate(ALL_CATEGORIES):
        positive_count = np.sum(labels[:, i])
        print(f"  {category}: {positive_count}/{len(labels)} ({positive_count/len(labels)*100:.1f}%)")
    
    return texts, labels, ALL_CATEGORIES

def create_model_and_tokenizer(num_labels):
    """Create RoBERTa model and tokenizer"""
    model_name = "roberta-base"
    
    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    
    # Load model for sequence classification
    model = RobertaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )
    
    return model, tokenizer

def train_model(model, tokenizer, train_texts, train_labels, val_texts, val_labels, 
                epochs=5, batch_size=16, learning_rate=2e-5, max_length=256, 
                patience=3, device='cpu', source='unknown'):
    """Train the RoBERTa model"""
    
    # Create datasets
    train_dataset = HomelessnessDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = HomelessnessDataset(val_texts, val_labels, tokenizer, max_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Move model to device
    model = model.to(device)
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps), 
        num_training_steps=total_steps
    )
    
    # Loss function for multi-label classification
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    best_macro_f1 = 0
    patience_counter = 0
    
    print("Training model...")
    print(f"Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Training
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} (Train)')
        for batch in train_pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            loss = criterion(logits, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation
        model.eval()
        val_predictions = []
        val_true_labels = []
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} (Val)')
        with torch.no_grad():
            for batch in val_pbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(logits)
                predictions = (probs >= 0.5).float()
                
                val_predictions.append(predictions.cpu().numpy())
                val_true_labels.append(labels.cpu().numpy())
        
        # Calculate metrics
        val_predictions = np.vstack(val_predictions)
        val_true_labels = np.vstack(val_true_labels)
        
        # Find best threshold for macro F1
        best_threshold = 0.5
        best_macro_f1_threshold = 0
        
        for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
            preds_threshold = (val_predictions >= threshold).astype(int)
            macro_f1 = f1_score(val_true_labels, preds_threshold, average='macro', zero_division=0)
            if macro_f1 > best_macro_f1_threshold:
                best_macro_f1_threshold = macro_f1
                best_threshold = threshold
        
        # Calculate final metrics with best threshold
        final_preds = (val_predictions >= best_threshold).astype(int)
        macro_f1 = f1_score(val_true_labels, final_preds, average='macro', zero_division=0)
        micro_f1 = f1_score(val_true_labels, final_preds, average='micro', zero_division=0)
        
        avg_loss = total_loss / len(train_loader)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_loss:.4f}, Val Macro F1: {macro_f1:.4f}, Val Micro F1: {micro_f1:.4f}, Best Threshold: {best_threshold}")
        
        # Early stopping
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            patience_counter = 0
            # Save best model
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), f'models/roberta_best_{source}.pt')
            print(f"New best model saved! Macro F1: {macro_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    return model, best_threshold

def evaluate_model(model, tokenizer, test_texts, test_labels, threshold=0.5, 
                  device='cpu', max_length=256):
    """Evaluate the RoBERTa model"""
    
    # Create test dataset
    test_dataset = HomelessnessDataset(test_texts, test_labels, tokenizer, max_length)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    model = model.to(device)
    model.eval()
    
    predictions = []
    true_labels = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()
            
            predictions.append(preds.cpu().numpy())
            true_labels.append(labels.cpu().numpy())
    
    predictions = np.vstack(predictions)
    true_labels = np.vstack(true_labels)
    
    # Calculate metrics
    macro_f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)
    micro_f1 = f1_score(true_labels, predictions, average='micro', zero_division=0)
    
    # Per-category metrics
    category_metrics = {}
    for i, category in enumerate(ALL_CATEGORIES):
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels[:, i], predictions[:, i], average='binary', zero_division=0
        )
        category_metrics[category] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'threshold': threshold,
        'category_metrics': category_metrics,
        'predictions': predictions,
        'true_labels': true_labels
    }

def save_results(results, source):
    """Save evaluation results"""
    os.makedirs(f'output/{source}/roberta', exist_ok=True)
    
    # Save metrics
    metrics = {
        'macro_f1': results['macro_f1'],
        'micro_f1': results['micro_f1'],
        'threshold': results['threshold'],
        'category_metrics': results['category_metrics']
    }
    
    with open(f'output/{source}/roberta/roberta_metrics_{source}.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save per-category results
    category_results = []
    for category, metrics in results['category_metrics'].items():
        category_results.append({
            'category': category,
            'f1': metrics['f1'],
            'precision': metrics['precision'],
            'recall': metrics['recall']
        })
    
    df = pd.DataFrame(category_results)
    df.to_csv(f'output/{source}/roberta/roberta_classification_results_{source}.csv', index=False)
    
    print(f"\nResults saved to output/{source}/roberta/")
    print(f"Macro F1: {results['macro_f1']:.4f}")
    print(f"Micro F1: {results['micro_f1']:.4f}")
    print(f"Best threshold: {results['threshold']}")

def predict_on_data(model, tokenizer, texts, threshold=0.5, device='cpu', max_length=256):
    """Make predictions on new data"""
    
    # Create dataset
    dummy_labels = np.zeros((len(texts), len(ALL_CATEGORIES)))
    dataset = HomelessnessDataset(texts, dummy_labels, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    model = model.to(device)
    model.eval()
    
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Making predictions"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()
            
            predictions.append(preds.cpu().numpy())
            probabilities.append(probs.cpu().numpy())
    
    predictions = np.vstack(predictions)
    probabilities = np.vstack(probabilities)
    
    return predictions, probabilities

def main():
    parser = argparse.ArgumentParser(description='RoBERTa Multiclass Classifier')
    parser.add_argument('--source', required=True, choices=['reddit', 'x', 'news', 'meeting_minutes'],
                       help='Data source to use')
    parser.add_argument('--mode', required=True, choices=['train', 'evaluate', 'predict'],
                       help='Mode to run')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--val_size', type=float, default=0.1, help='Validation set size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--input', help='Input file for prediction mode')
    parser.add_argument('--output', help='Output file for prediction mode')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    texts, labels, categories = load_and_preprocess_data(args.source)
    
    # Split data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=args.test_size, random_state=args.seed, stratify=None
    )
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=args.val_size/(1-args.test_size), 
        random_state=args.seed, stratify=None
    )
    
    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
    
    # Create model and tokenizer
    model, tokenizer = create_model_and_tokenizer(len(ALL_CATEGORIES))
    
    if args.mode == 'train':
        # Train model
        model, best_threshold = train_model(
            model, tokenizer, train_texts, train_labels, val_texts, val_labels,
            epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
            max_length=args.max_length, device=device, source=args.source
        )
        
        # Evaluate on test set
        print("Loading best model for evaluation")
        model.load_state_dict(torch.load(f'models/roberta_best_{args.source}.pt'))
        results = evaluate_model(model, tokenizer, test_texts, test_labels, 
                               threshold=best_threshold, device=device, max_length=args.max_length)
        save_results(results, args.source)
        
    elif args.mode == 'evaluate':
        # Load trained model
        model_path = f'models/roberta_best_{args.source}.pt'
        if not os.path.exists(model_path):
            print(f"Trained model not found: {model_path}")
            return
        
        model.load_state_dict(torch.load(model_path))
        results = evaluate_model(model, tokenizer, test_texts, test_labels, 
                               device=device, max_length=args.max_length)
        save_results(results, args.source)
        
    elif args.mode == 'predict':
        if not args.input or not args.output:
            print("Please provide --input and --output files for prediction mode")
            return
        
        # Load trained model
        model_path = f'models/roberta_best_{args.source}.pt'
        if not os.path.exists(model_path):
            print(f"Trained model not found: {model_path}")
            return
        
        model.load_state_dict(torch.load(model_path))
        
        # Load input data
        df = pd.read_csv(args.input)
        texts = df.iloc[:, 0].fillna('').astype(str).tolist()  # Assume first column is text
        
        # Make predictions
        predictions, probabilities = predict_on_data(model, tokenizer, texts, device=device, max_length=args.max_length)
        
        # Save results
        results_df = pd.DataFrame(predictions, columns=ALL_CATEGORIES)
        results_df.insert(0, 'text', texts)
        results_df.to_csv(args.output, index=False)
        
        print(f"Predictions saved to {args.output}")

if __name__ == "__main__":
    main() 