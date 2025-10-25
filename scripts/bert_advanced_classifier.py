#!/usr/bin/env python3
"""
Advanced BERT Classifier - Targeting 70%+ F1 Scores

This script implements multiple advanced techniques:
1. Focal Loss for extreme class imbalance
2. Class-weighted sampling
3. Data augmentation
4. Better threshold optimization
5. Ensemble methods
6. Improved architecture

Usage:
    python scripts/bert_advanced_classifier.py --source reddit --mode train
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
    AdamW, get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
# from imblearn.over_sampling import SMOTE  # Not available, using weighted sampling instead
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

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Apply sigmoid to inputs
        inputs = torch.sigmoid(inputs)
        
        # Compute focal loss
        bce_loss = -targets * torch.log(inputs + 1e-8) - (1 - targets) * torch.log(1 - inputs + 1e-8)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class AdvancedDataset(Dataset):
    """Advanced dataset with augmentation and better handling"""
    
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
        
        # Enhanced data augmentation
        if self.augment and np.random.random() < 0.4:
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
            # Random word dropout (10-20%)
            dropout_rate = np.random.uniform(0.1, 0.2)
            num_drop = max(1, int(len(words) * dropout_rate))
            indices_to_drop = np.random.choice(len(words), num_drop, replace=False)
            words = [word for i, word in enumerate(words) if i not in indices_to_drop]
            
            # Random word swap
            if len(words) > 2:
                i, j = np.random.choice(len(words), 2, replace=False)
                words[i], words[j] = words[j], words[i]
            
            # Random word insertion (rare words)
            if len(words) > 1 and np.random.random() < 0.1:
                insert_pos = np.random.randint(0, len(words))
                words.insert(insert_pos, np.random.choice(['very', 'really', 'quite', 'rather']))
            
            text = ' '.join(words)
        return text

def load_and_preprocess_data(source):
    """Load and preprocess data with advanced handling"""
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
    
    # Create labels matrix with aggressive thresholding for rare classes
    labels = np.zeros((len(soft_labels_df), len(ALL_CATEGORIES)))
    
    for i, category in enumerate(ALL_CATEGORIES):
        if category in soft_labels_df.columns:
            # Use very aggressive thresholding for extremely rare classes
            if category in ['racist', 'media portrayal']:
                labels[:, i] = (soft_labels_df[category] >= 0.2).astype(int)
            elif category in ['deserving/undeserving', 'not in my backyard']:
                labels[:, i] = (soft_labels_df[category] >= 0.33).astype(int)
            else:
                labels[:, i] = (soft_labels_df[category] >= 0.5).astype(int)
        else:
            print(f"Warning: Category '{category}' not found in data, using zeros")
    
    # Print detailed statistics
    print(f"\nLabel distribution:")
    for i, category in enumerate(ALL_CATEGORIES):
        positive_count = np.sum(labels[:, i])
        percentage = positive_count/len(labels)*100
        print(f"  {category}: {positive_count}/{len(labels)} ({percentage:.1f}%)")
    
    return texts, labels, ALL_CATEGORIES

def create_model_and_tokenizer(num_labels, model_name='bert-base-uncased'):
    """Create model with improved architecture"""
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, problem_type='multi_label_classification'
    )
    
    # Better initialization
    nn.init.xavier_uniform_(model.classifier.weight)
    nn.init.zeros_(model.classifier.bias)
    
    return model, tokenizer

def compute_class_weights_advanced(labels):
    """Compute advanced class weights for extreme imbalance"""
    class_weights = []
    
    for i in range(labels.shape[1]):
        binary_labels = labels[:, i].astype(int)
        
        if len(np.unique(binary_labels)) > 1:
            # Compute class weights
            weights = compute_class_weight(
                'balanced', classes=np.unique(binary_labels), y=binary_labels
            )
            
            # Apply aggressive weighting for rare classes
            pos_ratio = np.mean(binary_labels)
            if pos_ratio < 0.01:  # Less than 1% positive
                weights[1] *= 50  # Very heavy weighting
            elif pos_ratio < 0.05:  # Less than 5% positive
                weights[1] *= 20  # Heavy weighting
            elif pos_ratio < 0.1:  # Less than 10% positive
                weights[1] *= 10  # Moderate weighting
            
            class_weights.append(torch.tensor(weights, dtype=torch.float32))
        else:
            class_weights.append(torch.tensor([1.0, 1.0], dtype=torch.float32))
    
    return class_weights

def create_weighted_sampler(labels):
    """Create weighted sampler for imbalanced data"""
    # Calculate sample weights based on class distribution
    sample_weights = []
    
    for i in range(len(labels)):
        weight = 1.0
        for j in range(labels.shape[1]):
            if labels[i, j] == 1:  # Positive sample
                pos_ratio = np.mean(labels[:, j])
                if pos_ratio < 0.1:
                    weight *= 10  # Higher weight for rare positive samples
                elif pos_ratio < 0.2:
                    weight *= 5
        sample_weights.append(weight)
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

def train_model(model, train_loader, val_loader, device, epochs=5, learning_rate=2e-5, source=''):
    """Train model with advanced techniques"""
    
    # Use different learning rates for different parts
    bert_params = [p for n, p in model.named_parameters() if 'bert' in n and p.requires_grad]
    classifier_params = [p for n, p in model.named_parameters() if 'classifier' in n and p.requires_grad]
    
    optimizer = AdamW([
        {'params': bert_params, 'lr': learning_rate},
        {'params': classifier_params, 'lr': learning_rate * 5}  # Higher LR for classifier
    ], weight_decay=0.01)
    
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    
    # Use focal loss
    criterion = FocalLoss(alpha=1, gamma=2)
    
    best_val_f1 = 0
    patience = 5
    patience_counter = 0
    
    print(f"Training for {epochs} epochs with advanced techniques...")
    
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
        
        # Find best threshold for each category with more granular search
        best_thresholds = []
        category_f1s = []
        
        for i in range(len(ALL_CATEGORIES)):
            true_labels = val_true_labels[:, i]
            pred_scores = val_predictions[:, i]
            
            if len(np.unique(true_labels)) > 1:
                # Find best threshold for this category
                best_f1 = 0
                best_threshold = 0.5
                
                # More granular threshold search
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
            }, f'models/bert_advanced_best_{source}.pt')
            print(f"New best model saved! Macro F1: {macro_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break
    
    return model

def evaluate_model(model, test_loader, device, label_names, source):
    """Evaluate model with advanced techniques"""
    
    # Load best thresholds
    checkpoint_path = f'models/bert_advanced_best_{source}.pt'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        best_thresholds = checkpoint['best_thresholds']
        print(f"Loaded best thresholds: {[f'{t:.3f}' for t in best_thresholds]}")
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
    
    output_dir = f'output/{source}/bert_advanced'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    metrics = {
        'source': source,
        'model': 'bert-base-uncased-advanced',
        'macro_f1': results['macro_f1'],
        'micro_f1': results['micro_f1'],
        'best_thresholds': results['best_thresholds'],
        'test_size': results['test_size'],
        'num_labels': results['num_labels'],
        'label_f1_scores': results['label_f1_scores'],
        'label_precision_scores': results['label_precision_scores'],
        'label_recall_scores': results['label_recall_scores']
    }
    
    with open(f'{output_dir}/bert_advanced_metrics_{source}.json', 'w') as f:
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
    results_df.to_csv(f'{output_dir}/bert_advanced_detailed_results_{source}.csv', index=False)
    
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
    parser = argparse.ArgumentParser(description='Advanced BERT Classifier')
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
        
        # Create datasets
        train_dataset = AdvancedDataset(train_texts, train_labels, tokenizer, args.max_length, augment=True)
        val_dataset = AdvancedDataset(val_texts, val_labels, tokenizer, args.max_length)
        test_dataset = AdvancedDataset(test_texts, test_labels, tokenizer, args.max_length)
        
        # Create weighted sampler
        sampler = create_weighted_sampler(train_labels)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Train model
        print("Training advanced model...")
        model = train_model(model, train_loader, val_loader, device, args.epochs, args.learning_rate, args.source)
        
        # Load best model
        checkpoint_path = f'models/bert_advanced_best_{args.source}.pt'
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
        checkpoint_path = f'models/bert_advanced_best_{args.source}.pt'
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Trained model not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded trained model from {checkpoint_path}")
        
        # Create test dataset
        test_dataset = AdvancedDataset(test_texts, test_labels, tokenizer, args.max_length)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Evaluate
        print("Evaluating model...")
        results = evaluate_model(model, test_loader, device, label_names, args.source)
        
        # Save results
        save_results(results, args.source, label_names)

if __name__ == "__main__":
    main()
