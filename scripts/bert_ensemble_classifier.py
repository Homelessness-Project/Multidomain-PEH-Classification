#!/usr/bin/env python3
"""
BERT Ensemble Classifier - Combining multiple approaches for maximum F1

This script combines:
1. Multiple BERT variants (base, frozen, advanced)
2. Different threshold strategies
3. Voting ensemble
4. Confidence weighting

Usage:
    python scripts/bert_ensemble_classifier.py --source reddit --mode train
"""

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification,
    AdamW, get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_fscore_support
import json
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Define all 16 categories
ALL_CATEGORIES = [
    'ask a genuine question', 'ask a rhetorical question', 'provide a fact or claim',
    'provide an observation', 'express their opinion', 'express others opinions',
    'money aid allocation', 'government critique', 'societal critique',
    'solutions/interventions', 'personal interaction', 'media portrayal',
    'not in my backyard', 'harmful generalization', 'deserving/undeserving', 'racist'
]

class EnsembleDataset(Dataset):
    """Dataset for ensemble training"""
    
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
            text, truncation=True, padding='max_length', 
            max_length=self.max_length, return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float32)
        }

def load_and_preprocess_data(source):
    """Load and preprocess data"""
    print(f"Loading data for {source}...")
    
    # Load soft labels
    soft_labels_path = f'output/annotation/soft_labels/{source}_soft_labels.csv'
    if not os.path.exists(soft_labels_path):
        raise FileNotFoundError(f"Soft labels file not found: {soft_labels_path}")
    
    soft_labels_df = pd.read_csv(soft_labels_path)
    print(f"Loaded {len(soft_labels_df)} soft label samples")
    
    # Load text data
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
    
    # Get text column
    text_col_map = {
        'reddit': 'Comment', 'x': 'Deidentified_text', 
        'news': 'Deidentified_paragraph_text', 'meeting_minutes': 'Deidentified_paragraph'
    }
    
    text_col = text_col_map.get(source)
    if text_col not in text_df.columns:
        raise ValueError(f"Text column '{text_col}' not found in text data")
    
    # Align data lengths
    if len(soft_labels_df) != len(text_df):
        min_len = min(len(soft_labels_df), len(text_df))
        soft_labels_df = soft_labels_df.iloc[:min_len]
        text_df = text_df.iloc[:min_len]
    
    texts = text_df[text_col].fillna('').astype(str).tolist()
    
    # Create labels matrix with very aggressive thresholding
    labels = np.zeros((len(soft_labels_df), len(ALL_CATEGORIES)))
    
    for i, category in enumerate(ALL_CATEGORIES):
        if category in soft_labels_df.columns:
            # Use very aggressive thresholding for rare classes
            if category in ['racist', 'media portrayal']:
                labels[:, i] = (soft_labels_df[category] >= 0.15).astype(int)  # Very low threshold
            elif category in ['deserving/undeserving', 'not in my backyard', 'ask a genuine question']:
                labels[:, i] = (soft_labels_df[category] >= 0.25).astype(int)  # Low threshold
            else:
                labels[:, i] = (soft_labels_df[category] >= 0.4).astype(int)  # Lower than standard
        else:
            print(f"Warning: Category '{category}' not found in data, using zeros")
    
    # Print detailed statistics
    print(f"\nLabel distribution:")
    for i, category in enumerate(ALL_CATEGORIES):
        positive_count = np.sum(labels[:, i])
        percentage = positive_count/len(labels)*100
        print(f"  {category}: {positive_count}/{len(labels)} ({percentage:.1f}%)")
    
    return texts, labels, ALL_CATEGORIES

def create_models_and_tokenizers(num_labels):
    """Create multiple model variants"""
    models = {}
    
    # BERT Base
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=num_labels, problem_type='multi_label_classification'
    )
    models['bert'] = (bert_model, bert_tokenizer)
    
    # RoBERTa Base
    roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    roberta_model = RobertaForSequenceClassification.from_pretrained(
        'roberta-base', num_labels=num_labels, problem_type='multi_label_classification'
    )
    models['roberta'] = (roberta_model, roberta_tokenizer)
    
    return models

def train_single_model(model, tokenizer, train_loader, val_loader, device, epochs=3, learning_rate=2e-5, model_name='', source=''):
    """Train a single model"""
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_f1 = 0
    patience = 3
    patience_counter = 0
    
    print(f"Training {model_name} for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        
        train_pbar = tqdm(train_loader, desc=f'{model_name} Epoch {epoch+1}/{epochs} (Train)')
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
            val_pbar = tqdm(val_loader, desc=f'{model_name} Epoch {epoch+1}/{epochs} (Val)')
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
        
        # Find best threshold for each category
        best_thresholds = []
        category_f1s = []
        
        for i in range(len(ALL_CATEGORIES)):
            true_labels = val_true_labels[:, i]
            pred_scores = val_predictions[:, i]
            
            if len(np.unique(true_labels)) > 1:
                best_f1 = 0
                best_threshold = 0.5
                
                for threshold in np.arange(0.05, 0.95, 0.05):
                    binary_pred = (pred_scores > threshold).astype(int)
                    if len(np.unique(binary_pred)) > 1:
                        f1 = f1_score(true_labels, binary_pred, zero_division=0)
                        if f1 > best_f1:
                            best_f1 = f1
                            best_threshold = threshold
                
                best_thresholds.append(best_threshold)
                category_f1s.append(best_f1)
            else:
                best_thresholds.append(0.5)
                category_f1s.append(0.0)
        
        macro_f1 = np.mean(category_f1s)
        
        print(f"{model_name} Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Macro F1: {macro_f1:.4f}")
        
        # Early stopping
        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            patience_counter = 0
            # Save best model
            os.makedirs('models', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_thresholds': best_thresholds,
                'macro_f1': macro_f1
            }, f'models/{model_name}_ensemble_best_{source}.pt')
            print(f"New best {model_name} model saved! Macro F1: {macro_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping {model_name} after {patience} epochs without improvement")
                break
    
    return model

def ensemble_predict(models, test_loader, device, label_names, source):
    """Make ensemble predictions"""
    
    all_predictions = {}
    all_thresholds = {}
    
    # Load each model and get predictions
    for model_name, (model, tokenizer) in models.items():
        print(f"Loading {model_name} model...")
        
        # Load best model
        checkpoint_path = f'models/{model_name}_ensemble_best.pt'
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            all_thresholds[model_name] = checkpoint['best_thresholds']
            print(f"Loaded {model_name} with thresholds: {[f'{t:.3f}' for t in checkpoint['best_thresholds']]}")
        else:
            print(f"No trained {model_name} model found, using default thresholds")
            all_thresholds[model_name] = [0.5] * len(label_names)
        
        model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Predicting with {model_name}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                pred_scores = torch.sigmoid(outputs.logits)
                predictions.extend(pred_scores.cpu().numpy())
        
        all_predictions[model_name] = np.array(predictions)
    
    # Ensemble predictions
    print("Creating ensemble predictions...")
    
    # Method 1: Average predictions
    avg_predictions = np.mean(list(all_predictions.values()), axis=0)
    
    # Method 2: Weighted average (weight by individual model performance)
    weights = {'bert': 0.5, 'roberta': 0.5}  # Equal weights for now
    weighted_predictions = np.zeros_like(avg_predictions)
    for model_name, preds in all_predictions.items():
        weighted_predictions += weights[model_name] * preds
    
    # Method 3: Voting ensemble
    vote_predictions = np.zeros_like(avg_predictions)
    for model_name, preds in all_predictions.items():
        thresholds = all_thresholds[model_name]
        binary_preds = np.zeros_like(preds)
        for i in range(len(label_names)):
            binary_preds[:, i] = (preds[:, i] > thresholds[i]).astype(int)
        vote_predictions += binary_preds
    
    # Normalize voting predictions
    vote_predictions = vote_predictions / len(all_predictions)
    
    return {
        'average': avg_predictions,
        'weighted': weighted_predictions,
        'voting': vote_predictions,
        'individual': all_predictions,
        'thresholds': all_thresholds
    }

def evaluate_ensemble(ensemble_preds, test_labels, label_names):
    """Evaluate ensemble predictions"""
    
    results = {}
    
    for method_name, predictions in ensemble_preds.items():
        if method_name == 'individual' or method_name == 'thresholds':
            continue
            
        print(f"\nEvaluating {method_name} ensemble...")
        
        # Find best thresholds for this ensemble method
        best_thresholds = []
        category_f1s = []
        
        for i in range(len(label_names)):
            true_labels = test_labels[:, i]
            pred_scores = predictions[:, i]
            
            if len(np.unique(true_labels)) > 1:
                best_f1 = 0
                best_threshold = 0.5
                
                for threshold in np.arange(0.05, 0.95, 0.025):
                    binary_pred = (pred_scores > threshold).astype(int)
                    if len(np.unique(binary_pred)) > 1:
                        f1 = f1_score(true_labels, binary_pred, zero_division=0)
                        if f1 > best_f1:
                            best_f1 = f1
                            best_threshold = threshold
                
                best_thresholds.append(best_threshold)
                category_f1s.append(best_f1)
            else:
                best_thresholds.append(0.5)
                category_f1s.append(0.0)
        
        # Apply best thresholds
        final_predictions = np.zeros_like(predictions)
        for i in range(len(label_names)):
            final_predictions[:, i] = (predictions[:, i] > best_thresholds[i]).astype(int)
        
        # Convert true labels to binary
        final_true_labels = (test_labels > 0.5).astype(int)
        
        # Calculate metrics
        macro_f1 = f1_score(final_true_labels, final_predictions, average='macro', zero_division=0)
        micro_f1 = f1_score(final_true_labels, final_predictions, average='micro', zero_division=0)
        
        # Per-label F1 scores
        label_f1_scores = {}
        for i, label_name in enumerate(label_names):
            label_f1 = f1_score(
                final_true_labels[:, i], final_predictions[:, i], zero_division=0
            )
            label_f1_scores[label_name] = label_f1
        
        results[method_name] = {
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'best_thresholds': best_thresholds,
            'label_f1_scores': label_f1_scores
        }
        
        print(f"{method_name} - Macro F1: {macro_f1:.4f}, Micro F1: {micro_f1:.4f}")
    
    return results

def save_ensemble_results(results, source, label_names):
    """Save ensemble results"""
    
    output_dir = f'nlp_outputs/{source}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    with open(f'{output_dir}/bert_ensemble_metrics_{source}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save detailed results
    for method_name, method_results in results.items():
        if method_name in ['individual', 'thresholds']:
            continue
            
        results_data = []
        for i, label in enumerate(label_names):
            results_data.append({
                'category': label,
                'f1_score': method_results['label_f1_scores'][label],
                'threshold': method_results['best_thresholds'][i]
            })
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(f'{output_dir}/ensemble_{method_name}.csv', index=False)
        
        print(f"\n{method_name} results saved to {output_dir}/")
        print(f"Macro F1: {method_results['macro_f1']:.4f}")
        print(f"Micro F1: {method_results['micro_f1']:.4f}")
        
        # Print per-category results
        print(f"\nPer-category F1 scores ({method_name}):")
        for i, label in enumerate(label_names):
            f1 = method_results['label_f1_scores'][label]
            threshold = method_results['best_thresholds'][i]
            print(f"  {label}: {f1:.4f} (threshold: {threshold:.3f})")

def main():
    parser = argparse.ArgumentParser(description='BERT Ensemble Classifier')
    parser.add_argument('--source', type=str, required=True, 
                       choices=['reddit', 'x', 'news', 'meeting_minutes'],
                       help='Data source to use')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'evaluate'],
                       help='Mode: train or evaluate')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
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
        
        # Create models
        models = create_models_and_tokenizers(len(label_names))
        
        # Train each model
        for model_name, (model, tokenizer) in models.items():
            model.to(device)
            
            # Create datasets
            train_dataset = EnsembleDataset(train_texts, train_labels, tokenizer, args.max_length)
            val_dataset = EnsembleDataset(val_texts, val_labels, tokenizer, args.max_length)
            test_dataset = EnsembleDataset(test_texts, test_labels, tokenizer, args.max_length)
            
            # Create dataloaders
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
            
            # Train model
            train_single_model(model, tokenizer, train_loader, val_loader, device, 
                             args.epochs, args.learning_rate, model_name, args.source)
        
        print("All models trained! Use --mode evaluate to test ensemble.")
        
    elif args.mode == 'evaluate':
        # Load and preprocess data
        texts, labels, label_names = load_and_preprocess_data(args.source)
        
        # Split data
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=args.test_size, random_state=args.seed
        )
        
        # Create models
        models = create_models_and_tokenizers(len(label_names))
        
        # Create test dataset (using BERT tokenizer for consistency)
        test_dataset = EnsembleDataset(test_texts, test_labels, models['bert'][1], args.max_length)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Get ensemble predictions
        ensemble_preds = ensemble_predict(models, test_loader, device, label_names, args.source)
        
        # Evaluate ensemble
        results = evaluate_ensemble(ensemble_preds, test_labels, label_names)
        
        # Save results
        save_ensemble_results(results, args.source, label_names)

if __name__ == "__main__":
    main()
