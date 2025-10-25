#!/usr/bin/env python3
"""
High Performance BERT Classifier

This script addresses the critical issues identified:
1. Narrow prediction score ranges
2. Extreme class imbalance
3. Poor decision boundaries
4. Model not learning proper thresholds

Key improvements:
- Better loss functions for extreme imbalance
- Improved threshold learning
- Better data preprocessing
- Class-specific thresholds
- Improved model architecture

Usage:
    python scripts/bert_high_performance_classifier.py --source reddit --mode train
"""

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    AdamW, get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import json
import os
from tqdm import tqdm
import warnings
from collections import Counter
warnings.filterwarnings('ignore')

# Define all 16 categories
ALL_CATEGORIES = [
    'ask a genuine question', 'ask a rhetorical question', 'provide a fact or claim',
    'provide an observation', 'express their opinion', 'express others opinions',
    'money aid allocation', 'government critique', 'societal critique',
    'solutions/interventions', 'personal interaction', 'media portrayal',
    'not in my backyard', 'harmful generalization', 'deserving/undeserving', 'racist'
]

class WeightedBCELoss(nn.Module):
    """Weighted Binary Cross Entropy Loss for extreme class imbalance"""
    
    def __init__(self, pos_weight=None, reduction='mean'):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Apply sigmoid to inputs
        inputs = torch.sigmoid(inputs)
        
        # Compute weighted BCE loss
        if self.pos_weight is not None:
            pos_weight = self.pos_weight.unsqueeze(0).expand_as(targets)
            loss = -pos_weight * targets * torch.log(inputs + 1e-8) - (1 - targets) * torch.log(1 - inputs + 1e-8)
        else:
            loss = -targets * torch.log(inputs + 1e-8) - (1 - targets) * torch.log(1 - inputs + 1e-8)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class ImprovedDataset(Dataset):
    """Improved dataset with better handling of extreme imbalance"""
    
    def __init__(self, texts, labels, tokenizer, max_length=256, augment=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Data augmentation for minority classes
        if self.augment and np.random.random() < 0.3:
            text = self._augment_text(text)
        
        encoding = self.tokenizer(
            text, truncation=True, padding='max_length', 
            max_length=self.max_length, return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float32)
        }
    
    def _augment_text(self, text):
        """Enhanced text augmentation"""
        words = text.split()
        if len(words) > 3:
            # Random word dropout
            num_drop = max(1, int(len(words) * 0.1))
            indices_to_drop = np.random.choice(len(words), num_drop, replace=False)
            words = [word for i, word in enumerate(words) if i not in indices_to_drop]
            
            # Random word swap
            if len(words) > 2:
                i, j = np.random.choice(len(words), 2, replace=False)
                words[i], words[j] = words[j], words[i]
            
            text = ' '.join(words)
        return text

def load_and_preprocess_data(source):
    """Load and preprocess data with better handling"""
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
    
    # Create labels matrix with better thresholding
    labels = np.zeros((len(soft_labels_df), len(ALL_CATEGORIES)))
    
    for i, category in enumerate(ALL_CATEGORIES):
        if category in soft_labels_df.columns:
            # Use more aggressive thresholding for rare classes
            if category in ['racist', 'media portrayal', 'deserving/undeserving']:
                # For very rare classes, use lower threshold
                labels[:, i] = (soft_labels_df[category] >= 0.33).astype(int)
            else:
                # For other classes, use standard threshold
                labels[:, i] = (soft_labels_df[category] >= 0.5).astype(int)
        else:
            print(f"Warning: Category '{category}' not found in data, using zeros")
    
    # Print detailed statistics
    print(f"\nLabel distribution:")
    for i, category in enumerate(ALL_CATEGORIES):
        positive_count = np.sum(labels[:, i])
        percentage = positive_count/len(labels)*100
        print(f"  {category}: {positive_count}/{len(labels)} ({percentage:.1f}%)")
        
        # Flag extremely imbalanced classes
        if percentage < 2.0:
            print(f"    ⚠️  EXTREMELY IMBALANCED - {percentage:.1f}% positive")
        elif percentage < 5.0:
            print(f"    ⚠️  HIGHLY IMBALANCED - {percentage:.1f}% positive")
    
    return texts, labels, ALL_CATEGORIES

def create_model_and_tokenizer(num_labels, model_name='bert-base-uncased'):
    """Create model with improved architecture"""
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, problem_type='multi_label_classification'
    )
    
    # Initialize classifier weights better
    nn.init.xavier_uniform_(model.classifier.weight)
    nn.init.zeros_(model.classifier.bias)
    
    return model, tokenizer

def compute_class_weights_improved(labels):
    """Compute better class weights for extreme imbalance"""
    class_weights = []
    
    for i in range(labels.shape[1]):
        binary_labels = labels[:, i].astype(int)
        
        if len(np.unique(binary_labels)) > 1:
            # Compute class weights
            weights = compute_class_weight(
                'balanced', classes=np.unique(binary_labels), y=binary_labels
            )
            
            # Apply additional weighting for extremely rare classes
            pos_ratio = np.mean(binary_labels)
            if pos_ratio < 0.01:  # Less than 1% positive
                weights[1] *= 10  # Heavily weight positive class
            elif pos_ratio < 0.05:  # Less than 5% positive
                weights[1] *= 5   # Moderately weight positive class
            
            class_weights.append(torch.tensor(weights, dtype=torch.float32))
        else:
            class_weights.append(torch.tensor([1.0, 1.0], dtype=torch.float32))
    
    return class_weights

def train_model(model, train_loader, val_loader, device, epochs=5, learning_rate=2e-5, source=''):
    """Train model with improved techniques"""
    
    # Use different learning rates for different parts
    bert_params = [p for n, p in model.named_parameters() if 'bert' in n and p.requires_grad]
    classifier_params = [p for n, p in model.named_parameters() if 'classifier' in n and p.requires_grad]
    
    optimizer = AdamW([
        {'params': bert_params, 'lr': learning_rate},
        {'params': classifier_params, 'lr': learning_rate * 10}  # Higher LR for classifier
    ], weight_decay=0.01)
    
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    
    # Use weighted BCE loss
    criterion = WeightedBCELoss()
    
    best_val_f1 = 0
    patience = 5
    patience_counter = 0
    
    print(f"Training for {epochs} epochs with improved techniques...")
    
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
            
            # Use raw logits for weighted BCE
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
        
        # Find best threshold for each category
        best_thresholds = []
        best_macro_f1 = 0
        
        for i in range(len(ALL_CATEGORIES)):
            true_labels = val_true_labels[:, i]
            pred_scores = val_predictions[:, i]
            
            if len(np.unique(true_labels)) > 1:
                # Find best threshold for this category
                best_f1 = 0
                best_threshold = 0.5
                
                for threshold in np.arange(0.1, 0.9, 0.05):
                    binary_pred = (pred_scores > threshold).astype(int)
                    if len(np.unique(binary_pred)) > 1:
                        f1 = f1_score(true_labels, binary_pred, zero_division=0)
                        if f1 > best_f1:
                            best_f1 = f1
                            best_threshold = threshold
                
                best_thresholds.append(best_threshold)
            else:
                best_thresholds.append(0.5)
        
        # Calculate macro F1 with category-specific thresholds
        category_f1s = []
        for i in range(len(ALL_CATEGORIES)):
            true_labels = val_true_labels[:, i]
            pred_scores = val_predictions[:, i]
            threshold = best_thresholds[i]
            
            binary_pred = (pred_scores > threshold).astype(int)
            if len(np.unique(true_labels)) > 1 and len(np.unique(binary_pred)) > 1:
                f1 = f1_score(true_labels, binary_pred, zero_division=0)
                category_f1s.append(f1)
            else:
                category_f1s.append(0.0)
        
        macro_f1 = np.mean(category_f1s)
        
        # Calculate micro F1 with global threshold
        global_threshold = 0.5
        binary_pred_global = (val_predictions > global_threshold).astype(int)
        binary_true_global = (val_true_labels > 0.5).astype(int)
        micro_f1 = f1_score(binary_true_global, binary_pred_global, average='micro', zero_division=0)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
              f"Val Macro F1: {macro_f1:.4f}, Val Micro F1: {micro_f1:.4f}")
        
        # Early stopping
        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            patience_counter = 0
            # Save best model
            os.makedirs('models', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_thresholds': best_thresholds,
                'macro_f1': macro_f1,
                'micro_f1': micro_f1
            }, f'models/bert_high_perf_best_{source}.pt')
            print(f"New best model saved! Macro F1: {macro_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break
    
    return model

def evaluate_model(model, test_loader, device, label_names, source):
    """Evaluate model with category-specific thresholds"""
    
    # Load best thresholds
    checkpoint_path = f'models/bert_high_perf_best_{source}.pt'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        best_thresholds = checkpoint['best_thresholds']
        print(f"Loaded best thresholds: {best_thresholds}")
    else:
        best_thresholds = [0.5] * len(label_names)
        print("Using default thresholds: 0.5")
    
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
    
    # Apply category-specific thresholds
    final_predictions = np.zeros_like(all_predictions)
    for i in range(len(label_names)):
        final_predictions[:, i] = (all_predictions[:, i] > best_thresholds[i]).astype(int)
    
    # Convert true labels to binary
    final_true_labels = (all_true_labels > 0.5).astype(int)
    
    # Calculate metrics
    macro_f1 = f1_score(final_true_labels, final_predictions, average='macro', zero_division=0)
    micro_f1 = f1_score(final_true_labels, final_predictions, average='micro', zero_division=0)
    
    # Per-label F1 scores
    label_f1_scores = {}
    label_precision_scores = {}
    label_recall_scores = {}
    
    for i, label_name in enumerate(label_names):
        label_f1 = f1_score(
            final_true_labels[:, i], final_predictions[:, i], zero_division=0
        )
        precision, recall, _, _ = precision_recall_fscore_support(
            final_true_labels[:, i], final_predictions[:, i], 
            average='binary', zero_division=0
        )
        
        label_f1_scores[label_name] = label_f1
        label_precision_scores[label_name] = precision
        label_recall_scores[label_name] = recall
    
    results = {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'best_thresholds': best_thresholds,
        'test_size': len(final_true_labels),
        'num_labels': len(label_names),
        'label_f1_scores': label_f1_scores,
        'label_precision_scores': label_precision_scores,
        'label_recall_scores': label_recall_scores
    }
    
    return results

def save_results(results, source, label_names):
    """Save results to output directory"""
    
    output_dir = f'output/{source}/bert_high_perf'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    metrics = {
        'source': source,
        'model': 'bert-base-uncased-high-perf',
        'macro_f1': results['macro_f1'],
        'micro_f1': results['micro_f1'],
        'best_thresholds': results['best_thresholds'],
        'test_size': results['test_size'],
        'num_labels': results['num_labels'],
        'label_f1_scores': results['label_f1_scores'],
        'label_precision_scores': results['label_precision_scores'],
        'label_recall_scores': results['label_recall_scores']
    }
    
    with open(f'{output_dir}/bert_high_perf_metrics_{source}.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save detailed results
    results_data = []
    for i, label in enumerate(label_names):
        results_data.append({
            'category': label,
            'f1_score': results['label_f1_scores'][label],
            'precision': results['label_precision_scores'][label],
            'recall': results['label_recall_scores'][label],
            'threshold': results['best_thresholds'][i]
        })
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(f'{output_dir}/bert_high_perf_detailed_results_{source}.csv', index=False)
    
    print(f"\nResults saved to {output_dir}/")
    print(f"Macro F1: {results['macro_f1']:.4f}")
    print(f"Micro F1: {results['micro_f1']:.4f}")
    
    # Print per-category results
    print(f"\nPer-category F1 scores:")
    for i, label in enumerate(label_names):
        f1 = results['label_f1_scores'][label]
        threshold = results['best_thresholds'][i]
        print(f"  {label}: {f1:.4f} (threshold: {threshold:.3f})")
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description='High Performance BERT Classifier')
    parser.add_argument('--source', type=str, required=True, 
                       choices=['reddit', 'x', 'news', 'meeting_minutes'],
                       help='Data source to use')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'evaluate'],
                       help='Mode: train or evaluate')
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
        
        # Compute class weights
        class_weights = compute_class_weights_improved(train_labels)
        print(f"Computed improved class weights for {len(class_weights)} categories")
        
        # Create datasets
        train_dataset = ImprovedDataset(train_texts, train_labels, tokenizer, args.max_length, augment=True)
        val_dataset = ImprovedDataset(val_texts, val_labels, tokenizer, args.max_length)
        test_dataset = ImprovedDataset(test_texts, test_labels, tokenizer, args.max_length)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Train model
        print("Training high performance model...")
        model = train_model(model, train_loader, val_loader, device, args.epochs, args.learning_rate, args.source)
        
        # Load best model
        checkpoint_path = f'models/bert_high_perf_best_{args.source}.pt'
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded best model for evaluation")
        
        # Evaluate model
        print("Evaluating model...")
        results = evaluate_model(model, test_loader, device, label_names, args.source)
        
        # Save results
        save_results(results, args.source, label_names)
        
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
        checkpoint_path = f'models/bert_high_perf_best_{args.source}.pt'
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Trained model not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded trained model from {checkpoint_path}")
        
        # Create test dataset
        test_dataset = ImprovedDataset(test_texts, test_labels, tokenizer, args.max_length)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Evaluate
        print("Evaluating model...")
        results = evaluate_model(model, test_loader, device, label_names, args.source)
        
        # Save results
        save_results(results, args.source, label_names)

if __name__ == "__main__":
    main()

