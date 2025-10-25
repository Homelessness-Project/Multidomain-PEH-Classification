#!/usr/bin/env python3
"""
BERT Frozen Layer Percentage Analysis

This script tests different frozen layer percentages to find the optimal balance
between performance and efficiency for BERT fine-tuning.

Frozen percentages tested:
- 0% (full fine-tuning)
- 25% (freeze first 3 layers)
- 50% (freeze first 6 layers) 
- 75% (freeze first 9 layers)
- 90% (freeze first 11 layers)
- 99% (freeze all but last layer)
- 99.9% (freeze all but classifier head)

Usage:
    python scripts/bert_frozen_percentage_analysis.py --source reddit --epochs 3
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
from sklearn.metrics import f1_score, precision_recall_fscore_support
import json
import os
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
warnings.filterwarnings('ignore')

# Define all 16 categories
ALL_CATEGORIES = [
    'ask a genuine question', 'ask a rhetorical question', 'provide a fact or claim',
    'provide an observation', 'express their opinion', 'express others opinions',
    'money aid allocation', 'government critique', 'societal critique',
    'solutions/interventions', 'personal interaction', 'media portrayal',
    'not in my backyard', 'harmful generalization', 'deserving/undeserving', 'racist'
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
            text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float32)
        }

def load_and_preprocess_data(source):
    """Load and preprocess data for a specific source"""
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
    
    # Create labels matrix
    labels = np.zeros((len(soft_labels_df), len(ALL_CATEGORIES)))
    
    for i, category in enumerate(ALL_CATEGORIES):
        if category in soft_labels_df.columns:
            labels[:, i] = (soft_labels_df[category] >= 0.5).astype(int)
        else:
            print(f"Warning: Category '{category}' not found in data, using zeros")
    
    return texts, labels, ALL_CATEGORIES

def create_model_with_frozen_percentage(num_labels, frozen_percentage, model_name='bert-base-uncased'):
    """Create BERT model with specified frozen percentage"""
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, problem_type='multi_label_classification'
    )
    
    # Calculate how many layers to freeze
    total_layers = len(model.bert.encoder.layer)
    layers_to_freeze = int(total_layers * frozen_percentage / 100)
    
    print(f"Freezing {layers_to_freeze}/{total_layers} layers ({frozen_percentage}%)")
    
    # Freeze specified number of layers
    for i in range(layers_to_freeze):
        for param in model.bert.encoder.layer[i].parameters():
            param.requires_grad = False
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {frozen_params:,}")
    print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    
    return model, tokenizer, {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': frozen_params,
        'trainable_percentage': 100 * trainable_params / total_params,
        'layers_frozen': layers_to_freeze,
        'total_layers': total_layers
    }

def train_model(model, train_loader, val_loader, device, epochs=3, learning_rate=2e-5):
    """Train the BERT model with early stopping"""
    
    # Only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=learning_rate)
    
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    
    best_val_f1 = 0
    patience = 3
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} (Train)'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_predictions = []
        val_true_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
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
        
        # Find best threshold
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
            val_true_labels, average='micro', zero_division=0
        )
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
              f"Val Macro F1: {best_macro_f1:.4f}, Val Micro F1: {micro_f1:.4f}")
        
        # Early stopping
        if best_macro_f1 > best_val_f1:
            best_val_f1 = best_macro_f1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break
    
    return model

def evaluate_model(model, test_loader, device, label_names):
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
    
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        binary_pred = (all_predictions > threshold).astype(int)
        macro_f1 = f1_score(all_true_labels, binary_pred, average='macro', zero_division=0)
        
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_threshold = threshold
    
    # Final evaluation
    final_predictions = (all_predictions > best_threshold).astype(int)
    final_true_labels = all_true_labels
    
    macro_f1 = f1_score(final_true_labels, final_predictions, average='macro', zero_division=0)
    micro_f1 = f1_score(final_true_labels, final_predictions, average='micro', zero_division=0)
    
    # Per-label F1 scores
    label_f1_scores = {}
    for i, label_name in enumerate(label_names):
        label_f1 = f1_score(
            final_true_labels[:, i], final_predictions[:, i], zero_division=0
        )
        label_f1_scores[label_name] = label_f1
    
    return {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'best_threshold': best_threshold,
        'test_size': len(final_true_labels),
        'label_f1_scores': label_f1_scores
    }

def run_frozen_percentage_analysis(source, frozen_percentages, epochs=3):
    """Run analysis for different frozen percentages"""
    
    print(f"\n{'='*80}")
    print(f"BERT FROZEN PERCENTAGE ANALYSIS - {source.upper()}")
    print(f"{'='*80}")
    
    # Load data
    texts, labels, label_names = load_and_preprocess_data(source)
    
    # Split data
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=42
    )
    
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42
    )
    
    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []
    
    for frozen_pct in frozen_percentages:
        print(f"\n{'='*60}")
        print(f"TESTING {frozen_pct}% FROZEN LAYERS")
        print(f"{'='*60}")
        
        # Create model
        model, tokenizer, param_info = create_model_with_frozen_percentage(
            len(label_names), frozen_pct
        )
        model.to(device)
        
        # Create datasets
        train_dataset = HomelessnessDataset(train_texts, train_labels, tokenizer)
        val_dataset = HomelessnessDataset(val_texts, val_labels, tokenizer)
        test_dataset = HomelessnessDataset(test_texts, test_labels, tokenizer)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # Train model
        model = train_model(model, train_loader, val_loader, device, epochs)
        
        # Evaluate model
        eval_results = evaluate_model(model, test_loader, device, label_names)
        
        # Store results
        result = {
            'frozen_percentage': frozen_pct,
            'macro_f1': eval_results['macro_f1'],
            'micro_f1': eval_results['micro_f1'],
            'best_threshold': eval_results['best_threshold'],
            'test_size': eval_results['test_size'],
            **param_info
        }
        
        results.append(result)
        
        print(f"Results: Macro F1: {eval_results['macro_f1']:.4f}, "
              f"Micro F1: {eval_results['micro_f1']:.4f}")
    
    return results

def create_visualizations(results, source):
    """Create visualizations for frozen percentage analysis"""
    
    os.makedirs('output/frozen_analysis', exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # 1. Performance vs Frozen Percentage
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(df['frozen_percentage'], df['macro_f1'], 'o-', label='Macro F1', linewidth=2)
    plt.plot(df['frozen_percentage'], df['micro_f1'], 's-', label='Micro F1', linewidth=2)
    plt.xlabel('Frozen Percentage (%)')
    plt.ylabel('F1 Score')
    plt.title(f'Performance vs Frozen Percentage - {source.title()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Trainable Parameters vs Performance
    plt.subplot(2, 2, 2)
    plt.scatter(df['trainable_percentage'], df['macro_f1'], s=100, alpha=0.7, label='Macro F1')
    plt.scatter(df['trainable_percentage'], df['micro_f1'], s=100, alpha=0.7, label='Micro F1')
    plt.xlabel('Trainable Parameters (%)')
    plt.ylabel('F1 Score')
    plt.title('Performance vs Trainable Parameters')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Layers Frozen vs Performance
    plt.subplot(2, 2, 3)
    plt.plot(df['layers_frozen'], df['macro_f1'], 'o-', label='Macro F1', linewidth=2)
    plt.plot(df['layers_frozen'], df['micro_f1'], 's-', label='Micro F1', linewidth=2)
    plt.xlabel('Layers Frozen')
    plt.ylabel('F1 Score')
    plt.title('Performance vs Layers Frozen')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Parameter Efficiency
    plt.subplot(2, 2, 4)
    efficiency = df['macro_f1'] / df['trainable_percentage'] * 100
    plt.bar(range(len(df)), efficiency, alpha=0.7)
    plt.xlabel('Frozen Percentage')
    plt.ylabel('Efficiency (F1 per % trainable)')
    plt.title('Parameter Efficiency')
    plt.xticks(range(len(df)), [f"{pct}%" for pct in df['frozen_percentage']])
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'output/frozen_analysis/bert_frozen_analysis_{source}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results
    df.to_csv(f'output/frozen_analysis/bert_frozen_results_{source}.csv', index=False)
    
    print(f"\nVisualizations and results saved to output/frozen_analysis/")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='BERT Frozen Percentage Analysis')
    parser.add_argument('--source', type=str, required=True, 
                       choices=['reddit', 'x', 'news', 'meeting_minutes'],
                       help='Data source to use')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--frozen_percentages', type=float, nargs='+', 
                       default=[0, 25, 50, 75, 90, 99, 99.9],
                       help='Frozen percentages to test')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    print(f"BERT FROZEN PERCENTAGE ANALYSIS")
    print(f"Source: {args.source}")
    print(f"Frozen percentages: {args.frozen_percentages}")
    print(f"Epochs: {args.epochs}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run analysis
    results = run_frozen_percentage_analysis(args.source, args.frozen_percentages, args.epochs)
    
    # Create visualizations
    df = create_visualizations(results, args.source)
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY RESULTS")
    print(f"{'='*80}")
    print(df[['frozen_percentage', 'layers_frozen', 'trainable_percentage', 
              'macro_f1', 'micro_f1']].to_string(index=False, float_format='%.4f'))
    
    # Find optimal frozen percentage
    best_macro_idx = df['macro_f1'].idxmax()
    best_micro_idx = df['micro_f1'].idxmax()
    
    print(f"\nBest Macro F1: {df.loc[best_macro_idx, 'macro_f1']:.4f} "
          f"at {df.loc[best_macro_idx, 'frozen_percentage']}% frozen")
    print(f"Best Micro F1: {df.loc[best_micro_idx, 'micro_f1']:.4f} "
          f"at {df.loc[best_micro_idx, 'frozen_percentage']}% frozen")
    
    print(f"\nAnalysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()

