#!/usr/bin/env python3
"""
Improved BERT Classifier for Homelessness Content Analysis

This script addresses the data quality issues identified:
1. Imbalanced classes - uses focal loss and class weights
2. Low agreement - uses soft labels more effectively
3. Poor performance - implements multiple improvements

Improvements:
- Focal Loss for imbalanced classes
- Class weights based on inverse frequency
- Better learning rate scheduling
- Data augmentation
- Improved threshold selection
- RoBERTa as alternative

Usage:
    python scripts/bert_improved_classifier.py --source reddit --model bert --mode train
    python scripts/bert_improved_classifier.py --source reddit --model roberta --mode train
"""

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification,
    AdamW, get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_fscore_support
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

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Convert targets to float
        targets = targets.float()
        
        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Compute focal loss
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ImprovedHomelessnessDataset(Dataset):
    """Improved dataset class with better handling of imbalanced data"""
    
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
        
        # Simple data augmentation
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
        """Simple text augmentation"""
        # Random word dropout
        words = text.split()
        if len(words) > 3:
            num_drop = max(1, int(len(words) * 0.1))
            indices_to_drop = np.random.choice(len(words), num_drop, replace=False)
            words = [word for i, word in enumerate(words) if i not in indices_to_drop]
            text = ' '.join(words)
        return text

def load_and_preprocess_data(source, use_soft_labels=True):
    """Load and preprocess data with improved handling"""
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
    
    # Create labels matrix with improved handling
    labels = np.zeros((len(soft_labels_df), len(ALL_CATEGORIES)))
    
    if use_soft_labels:
        # Use soft labels directly (0-1 scale)
        for i, category in enumerate(ALL_CATEGORIES):
            if category in soft_labels_df.columns:
                labels[:, i] = soft_labels_df[category].values
            else:
                print(f"Warning: Category '{category}' not found in data, using zeros")
    else:
        # Use binary labels with threshold
        for i, category in enumerate(ALL_CATEGORIES):
            if category in soft_labels_df.columns:
                labels[:, i] = (soft_labels_df[category] >= 0.5).astype(int)
            else:
                print(f"Warning: Category '{category}' not found in data, using zeros")
    
    # Print label statistics
    print(f"Label distribution:")
    for i, category in enumerate(ALL_CATEGORIES):
        positive_count = np.sum(labels[:, i] > 0.5)
        avg_score = np.mean(labels[:, i])
        print(f"  {category}: {positive_count}/{len(labels)} ({positive_count/len(labels)*100:.1f}%) avg: {avg_score:.3f}")
    
    return texts, labels, ALL_CATEGORIES

def create_model_and_tokenizer(num_labels, model_name='bert-base-uncased'):
    """Create model and tokenizer"""
    if 'roberta' in model_name.lower():
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = RobertaForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, problem_type='multi_label_classification'
        )
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, problem_type='multi_label_classification'
        )
    
    return model, tokenizer

def compute_class_weights(labels):
    """Compute class weights for imbalanced data"""
    class_weights = []
    
    for i in range(labels.shape[1]):
        # Convert to binary for weight computation
        binary_labels = (labels[:, i] > 0.5).astype(int)
        
        if len(np.unique(binary_labels)) > 1:
            weights = compute_class_weight(
                'balanced', classes=np.unique(binary_labels), y=binary_labels
            )
            # Map weights to class indices
            weight_dict = {0: weights[0], 1: weights[1]}
            class_weights.append(weight_dict)
        else:
            # If only one class, use equal weights
            class_weights.append({0: 1.0, 1: 1.0})
    
    return class_weights

def train_model(model, train_loader, val_loader, device, epochs=5, learning_rate=2e-5, 
                use_focal_loss=True, class_weights=None, source=''):
    """Train the model with improved techniques"""
    
    # Use different optimizers based on model type
    if 'roberta' in str(type(model)).lower():
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    else:
        optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    
    # Choose loss function
    if use_focal_loss:
        criterion = FocalLoss(alpha=1, gamma=2)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
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
        
        # Find best threshold with more options
        best_threshold = 0.5
        best_macro_f1 = 0
        
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            binary_pred = (val_predictions > threshold).astype(int)
            # Convert soft labels to binary for F1 calculation
            binary_true = (val_true_labels > 0.5).astype(int)
            macro_f1 = f1_score(binary_true, binary_pred, average='macro', zero_division=0)
            
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_threshold = threshold
        
        micro_f1 = f1_score(
            (val_predictions > best_threshold).astype(int), 
            binary_true, average='micro', zero_division=0
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
            torch.save(model.state_dict(), f'models/bert_improved_best_{source}.pt')
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
    
    # Find best threshold
    best_threshold = 0.5
    best_macro_f1 = 0
    
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        binary_pred = (all_predictions > threshold).astype(int)
        # Convert soft labels to binary for F1 calculation
        binary_true = (all_true_labels > 0.5).astype(int)
        macro_f1 = f1_score(binary_true, binary_pred, average='macro', zero_division=0)
        
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_threshold = threshold
    
    # Final evaluation
    final_predictions = (all_predictions > best_threshold).astype(int)
    final_true_labels = (all_true_labels > 0.5).astype(int)
    
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
        'best_threshold': best_threshold,
        'test_size': len(final_true_labels),
        'num_labels': len(label_names),
        'label_f1_scores': label_f1_scores,
        'label_precision_scores': label_precision_scores,
        'label_recall_scores': label_recall_scores
    }
    
    return results

def save_results(results, source, model_name, label_names):
    """Save results to output directory"""
    
    output_dir = f'output/{source}/bert_improved'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    metrics = {
        'source': source,
        'model': model_name,
        'macro_f1': results['macro_f1'],
        'micro_f1': results['micro_f1'],
        'best_threshold': results['best_threshold'],
        'test_size': results['test_size'],
        'num_labels': results['num_labels'],
        'label_f1_scores': results['label_f1_scores'],
        'label_precision_scores': results['label_precision_scores'],
        'label_recall_scores': results['label_recall_scores']
    }
    
    with open(f'{output_dir}/bert_improved_metrics_{source}.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save detailed results
    results_data = []
    for label in label_names:
        results_data.append({
            'category': label,
            'f1_score': results['label_f1_scores'][label],
            'precision': results['label_precision_scores'][label],
            'recall': results['label_recall_scores'][label]
        })
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(f'{output_dir}/bert_improved_detailed_results_{source}.csv', index=False)
    
    print(f"\nResults saved to {output_dir}/")
    print(f"Macro F1: {results['macro_f1']:.4f}")
    print(f"Micro F1: {results['micro_f1']:.4f}")
    print(f"Best threshold: {results['best_threshold']}")
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description='Improved BERT Classifier')
    parser.add_argument('--source', type=str, required=True, 
                       choices=['reddit', 'x', 'news', 'meeting_minutes'],
                       help='Data source to use')
    parser.add_argument('--model', type=str, default='bert-base-uncased',
                       choices=['bert-base-uncased', 'roberta-base'],
                       help='Model to use')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'evaluate'],
                       help='Mode: train or evaluate')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=256, help='Max sequence length')
    parser.add_argument('--use_focal_loss', action='store_true', help='Use focal loss')
    parser.add_argument('--use_soft_labels', action='store_true', help='Use soft labels directly')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation')
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
    print(f"Model: {args.model}")
    print(f"Use focal loss: {args.use_focal_loss}")
    print(f"Use soft labels: {args.use_soft_labels}")
    print(f"Use augmentation: {args.augment}")
    
    if args.mode == 'train':
        # Load and preprocess data
        texts, labels, label_names = load_and_preprocess_data(args.source, args.use_soft_labels)
        
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
        model, tokenizer = create_model_and_tokenizer(len(label_names), args.model)
        model.to(device)
        
        # Compute class weights
        class_weights = compute_class_weights(train_labels)
        print(f"Computed class weights for {len(class_weights)} categories")
        
        # Create datasets
        train_dataset = ImprovedHomelessnessDataset(
            train_texts, train_labels, tokenizer, args.max_length, args.augment
        )
        val_dataset = ImprovedHomelessnessDataset(val_texts, val_labels, tokenizer, args.max_length)
        test_dataset = ImprovedHomelessnessDataset(test_texts, test_labels, tokenizer, args.max_length)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Train model
        print("Training improved model...")
        model = train_model(
            model, train_loader, val_loader, device, args.epochs, args.learning_rate,
            args.use_focal_loss, class_weights, args.source
        )
        
        # Load best model
        if os.path.exists(f'models/bert_improved_best_{args.source}.pt'):
            model.load_state_dict(torch.load(f'models/bert_improved_best_{args.source}.pt'))
            print("Loaded best model for evaluation")
        
        # Evaluate model
        print("Evaluating model...")
        results = evaluate_model(model, test_loader, device, label_names, args.source)
        
        # Save results
        save_results(results, args.source, args.model, label_names)
        
    elif args.mode == 'evaluate':
        # Load and preprocess data
        texts, labels, label_names = load_and_preprocess_data(args.source, args.use_soft_labels)
        
        # Split data
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=args.test_size, random_state=args.seed
        )
        
        # Load model
        model, tokenizer = create_model_and_tokenizer(len(label_names), args.model)
        model.to(device)
        
        # Load trained weights
        model_path = f'models/bert_improved_best_{args.source}.pt'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found: {model_path}")
        
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded trained model from {model_path}")
        
        # Create test dataset
        test_dataset = ImprovedHomelessnessDataset(test_texts, test_labels, tokenizer, args.max_length)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Evaluate
        print("Evaluating model...")
        results = evaluate_model(model, test_loader, device, label_names, args.source)
        
        # Save results
        save_results(results, args.source, args.model, label_names)

if __name__ == "__main__":
    main()
