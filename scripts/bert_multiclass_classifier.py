#!/usr/bin/env python3
"""
BERT Multiclass Classifier for Homelessness Content Analysis

This script provides a comprehensive BERT-based multiclass classification system
for analyzing homelessness-related content across 16 categories:

Comment Types (6):
- ask a genuine question
- ask a rhetorical question  
- provide a fact or claim
- provide an observation
- express their opinion
- express others opinions

Critique Categories (3):
- money aid allocation
- government critique
- societal critique

Response Categories (1):
- solutions/interventions

Perception Types (5):
- personal interaction
- media portrayal
- not in my backyard
- harmful generalization
- deserving/undeserving

Racist Classification (1):
- racist (Yes/No)

Usage:
    python scripts/bert_multiclass_classifier.py --source reddit --dataset gold_subset --mode train
    python scripts/bert_multiclass_classifier.py --source reddit --dataset gold_subset --mode predict --input data.csv
    python scripts/bert_multiclass_classifier.py --source reddit --dataset gold_subset --mode evaluate
"""

import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    AdamW, 
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, precision_recall_fscore_support
import json
import os
from tqdm import tqdm
import warnings
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
    """Dataset class for homelessness content classification"""
    
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
            'labels': torch.tensor(label, dtype=torch.float32)
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

def create_model_and_tokenizer(num_labels, model_name='bert-base-uncased'):
    """Create BERT model and tokenizer"""
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type='multi_label_classification'
    )
    return model, tokenizer

def train_model(model, train_loader, val_loader, device, epochs=5, learning_rate=2e-5):
    """Train the BERT model with early stopping"""
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=total_steps
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    
    best_val_f1 = 0
    patience = 5
    patience_counter = 0
    
    print(f"Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} (Train)')
        for batch in train_pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_predictions = []
        val_true_labels = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} (Val)')
            for batch in val_pbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.sigmoid(outputs.logits)
                
                val_predictions.extend(predictions.cpu().numpy())
                val_true_labels.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        val_predictions = np.array(val_predictions)
        val_true_labels = np.array(val_true_labels)
        
        # Find best threshold for macro F1
        best_threshold = 0.5
        best_macro_f1 = 0
        
        for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
            binary_pred = (val_predictions > threshold).astype(int)
            macro_f1 = f1_score(val_true_labels, binary_pred, average='macro', zero_division=0)
            
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_threshold = threshold
        
        micro_f1 = f1_score(
            (val_predictions > best_threshold).astype(int), 
            val_true_labels, 
            average='micro', 
            zero_division=0
        )
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
              f"Val Macro F1: {best_macro_f1:.4f}, Val Micro F1: {micro_f1:.4f}, "
              f"Best Threshold: {best_threshold}")
        
        # Early stopping
        if best_macro_f1 > best_val_f1:
            best_val_f1 = best_macro_f1
            patience_counter = 0
            # Save best model
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), f'models/bert_best_{source}.pt')
            print(f"New best model saved! Macro F1: {best_macro_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break
    
    return model

def evaluate_model(model, test_loader, device, label_names, source):
    """Evaluate the trained model"""
    
    model.eval()
    all_predictions = []
    all_true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.sigmoid(outputs.logits)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    
    # Find best threshold for macro F1
    best_threshold = 0.5
    best_macro_f1 = 0
    
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        binary_pred = (all_predictions > threshold).astype(int)
        macro_f1 = f1_score(all_true_labels, binary_pred, average='macro', zero_division=0)
        
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_threshold = threshold
    
    # Final evaluation with best threshold
    final_predictions = (all_predictions > best_threshold).astype(int)
    final_true_labels = all_true_labels
    
    macro_f1 = f1_score(final_true_labels, final_predictions, average='macro', zero_division=0)
    micro_f1 = f1_score(final_true_labels, final_predictions, average='micro', zero_division=0)
    
    # Per-label F1 scores
    label_f1_scores = {}
    for i, label_name in enumerate(label_names):
        label_f1 = f1_score(
            final_true_labels[:, i], 
            final_predictions[:, i], 
            zero_division=0
        )
        label_f1_scores[label_name] = label_f1
    
    results = {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'best_threshold': best_threshold,
        'test_size': len(final_true_labels),
        'num_labels': len(label_names),
        'label_f1_scores': label_f1_scores
    }
    
    return results

def save_results(results, source, label_names):
    """Save results to output directory"""
    
    output_dir = f'output/{source}/bert'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    metrics = {
        'source': source,
        'model': 'bert-base-uncased',
        'macro_f1': results['macro_f1'],
        'micro_f1': results['micro_f1'],
        'best_threshold': results['best_threshold'],
        'test_size': results['test_size'],
        'num_labels': results['num_labels'],
        'label_f1_scores': results['label_f1_scores']
    }
    
    with open(f'{output_dir}/bert_metrics_{source}.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save classification results
    results_df = pd.DataFrame({
        'label': label_names,
        'f1_score': [results['label_f1_scores'][label] for label in label_names]
    })
    results_df.to_csv(f'{output_dir}/bert_classification_results_{source}.csv', index=False)
    
    print(f"\nResults saved to {output_dir}/")
    print(f"Macro F1: {results['macro_f1']:.4f}")
    print(f"Micro F1: {results['micro_f1']:.4f}")
    print(f"Best threshold: {results['best_threshold']}")
    
    # Print per-category results
    print(f"\nPer-category F1 scores:")
    for label in label_names:
        f1 = results['label_f1_scores'][label]
        print(f"  {label}: {f1:.4f}")

def predict_on_data(model, tokenizer, input_file, output_file, source, max_length=256):
    """Predict on new data"""
    
    # Load data
    df = pd.read_csv(input_file)
    
    # Get text column based on source
    text_col_map = {
        'reddit': 'Comment',
        'x': 'Deidentified_text', 
        'news': 'Deidentified_paragraph_text',
        'meeting_minutes': 'Deidentified_paragraph'
    }
    
    text_col = text_col_map.get(source, 'Comment')
    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' not found in data")
    
    texts = df[text_col].fillna('').astype(str).tolist()
    
    # Create dataset
    dataset = HomelessnessDataset(texts, np.zeros((len(texts), len(ALL_CATEGORIES))), tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Predict
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.sigmoid(outputs.logits)
            
            all_predictions.extend(predictions.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    
    # Create output DataFrame
    output_df = df.copy()
    
    # Add prediction columns
    for i, category in enumerate(ALL_CATEGORIES):
        output_df[f'pred_{category}'] = all_predictions[:, i]
        output_df[f'pred_{category}_binary'] = (all_predictions[:, i] > 0.5).astype(int)
    
    # Save results
    output_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    
    return output_df

def main():
    parser = argparse.ArgumentParser(description='BERT Multiclass Classifier for Homelessness Content Analysis')
    parser.add_argument('--source', type=str, required=True, 
                       choices=['reddit', 'x', 'news', 'meeting_minutes'],
                       help='Data source to use')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['gold_subset', 'all'],
                       help='Dataset to use')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'predict', 'evaluate'],
                       help='Mode: train, predict, or evaluate')
    parser.add_argument('--input', type=str, default=None,
                       help='Input file for prediction mode')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for prediction mode')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=256, help='Max sequence length')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--val_size', type=float, default=0.1, help='Validation set size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.mode == 'train':
        # Load and preprocess data
        texts, labels, label_names = load_and_preprocess_data(args.source)
        
        # Split data
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels, test_size=args.test_size + args.val_size, random_state=args.seed
        )
        
        val_size_ratio = args.val_size / (args.test_size + args.val_size)
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels, test_size=1-val_size_ratio, random_state=args.seed
        )
        
        print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
        
        # Initialize model and tokenizer
        model, tokenizer = create_model_and_tokenizer(len(label_names))
        model.to(device)
        
        # Create datasets
        train_dataset = HomelessnessDataset(train_texts, train_labels, tokenizer, args.max_length)
        val_dataset = HomelessnessDataset(val_texts, val_labels, tokenizer, args.max_length)
        test_dataset = HomelessnessDataset(test_texts, test_labels, tokenizer, args.max_length)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Train model
        print("Training model...")
        model = train_model(model, train_loader, val_loader, device, args.epochs, args.learning_rate)
        
        # Load best model
        if os.path.exists(f'models/bert_best_{args.source}.pt'):
            model.load_state_dict(torch.load(f'models/bert_best_{args.source}.pt'))
            print("Loaded best model for evaluation")
        
        # Evaluate model
        print("Evaluating model...")
        results = evaluate_model(model, test_loader, device, label_names, args.source)
        
        # Save results
        save_results(results, args.source, label_names)
        
    elif args.mode == 'predict':
        if not args.input:
            raise ValueError("--input is required for predict mode")
        
        if not args.output:
            args.output = f'output/{args.source}/bert/predictions_{args.source}.csv'
        
        # Load model
        model, tokenizer = create_model_and_tokenizer(len(ALL_CATEGORIES))
        model.to(device)
        
        # Load trained weights
        model_path = f'models/bert_best_{args.source}.pt'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found: {model_path}")
        
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded trained model from {model_path}")
        
        # Predict
        predict_on_data(model, tokenizer, args.input, args.output, args.source, args.max_length)
        
    elif args.mode == 'evaluate':
        # Load and preprocess data
        texts, labels, label_names = load_and_preprocess_data(args.source)
        
        # Split data
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=args.test_size, random_state=args.seed
        )
        
        # Load model
        model, tokenizer = create_model_and_tokenizer(len(label_names))
        model.to(device)
        
        # Load trained weights
        model_path = f'models/bert_best_{args.source}.pt'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found: {model_path}")
        
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded trained model from {model_path}")
        
        # Create test dataset
        test_dataset = HomelessnessDataset(test_texts, test_labels, tokenizer, args.max_length)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Evaluate
        print("Evaluating model...")
        results = evaluate_model(model, test_loader, device, label_names, args.source)
        
        # Save results
        save_results(results, args.source, label_names)

if __name__ == "__main__":
    main() 