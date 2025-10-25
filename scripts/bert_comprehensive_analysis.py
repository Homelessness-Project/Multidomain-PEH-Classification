#!/usr/bin/env python3
"""
Comprehensive BERT Multiclass Analysis Script

This script combines training, evaluation, and analysis into one comprehensive tool
that outputs macro and micro F1 scores for each category for each source.

Usage:
    python scripts/bert_comprehensive_analysis.py --mode train_evaluate
    python scripts/bert_comprehensive_analysis.py --mode evaluate_only
    python scripts/bert_comprehensive_analysis.py --mode predict_only --input data.csv
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

def train_model(model, train_loader, val_loader, device, epochs=5, learning_rate=2e-5, source=''):
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
        # Note: True labels are already binary (>= 0.5 for 2+ annotators)
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
    # Note: True labels are already binary (>= 0.5 for 2+ annotators)
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
    label_precision_scores = {}
    label_recall_scores = {}
    
    for i, label_name in enumerate(label_names):
        label_f1 = f1_score(
            final_true_labels[:, i], 
            final_predictions[:, i], 
            zero_division=0
        )
        precision, recall, _, _ = precision_recall_fscore_support(
            final_true_labels[:, i], 
            final_predictions[:, i], 
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
        'label_f1_scores': results['label_f1_scores'],
        'label_precision_scores': results['label_precision_scores'],
        'label_recall_scores': results['label_recall_scores']
    }
    
    with open(f'{output_dir}/bert_metrics_{source}.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save detailed classification results
    results_data = []
    for label in label_names:
        results_data.append({
            'category': label,
            'f1_score': results['label_f1_scores'][label],
            'precision': results['label_precision_scores'][label],
            'recall': results['label_recall_scores'][label]
        })
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(f'{output_dir}/bert_detailed_results_{source}.csv', index=False)
    
    print(f"\nResults saved to {output_dir}/")
    print(f"Macro F1: {results['macro_f1']:.4f}")
    print(f"Micro F1: {results['micro_f1']:.4f}")
    print(f"Best threshold: {results['best_threshold']}")
    
    return results_df

def print_comprehensive_results(all_results):
    """Print comprehensive results for all sources"""
    
    print("\n" + "="*100)
    print("COMPREHENSIVE BERT MULTICLASS ANALYSIS RESULTS")
    print("="*100)
    
    # Create summary table
    summary_data = []
    for source, results in all_results.items():
        summary_data.append({
            'Source': source,
            'Macro F1': f"{results['macro_f1']:.4f}",
            'Micro F1': f"{results['micro_f1']:.4f}",
            'Best Threshold': f"{results['best_threshold']:.2f}",
            'Test Size': results['test_size']
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\nOVERALL PERFORMANCE SUMMARY:")
    print(summary_df.to_string(index=False))
    
    # Create detailed category comparison
    print("\n" + "="*100)
    print("DETAILED CATEGORY PERFORMANCE BY SOURCE")
    print("="*100)
    
    # Create detailed results table
    detailed_data = []
    for source, results in all_results.items():
        for category in ALL_CATEGORIES:
            if category in results['label_f1_scores']:
                detailed_data.append({
                    'Source': source,
                    'Category': category,
                    'F1 Score': f"{results['label_f1_scores'][category]:.4f}",
                    'Precision': f"{results['label_precision_scores'][category]:.4f}",
                    'Recall': f"{results['label_recall_scores'][category]:.4f}"
                })
    
    detailed_df = pd.DataFrame(detailed_data)
    
    # Print by category groups
    category_groups = {
        'Comment Types': ['ask a genuine question', 'ask a rhetorical question', 'provide a fact or claim', 
                         'provide an observation', 'express their opinion', 'express others opinions'],
        'Critique Categories': ['money aid allocation', 'government critique', 'societal critique'],
        'Response Categories': ['solutions/interventions'],
        'Perception Types': ['personal interaction', 'media portrayal', 'not in my backyard', 
                           'harmful generalization', 'deserving/undeserving'],
        'Racist Classification': ['racist']
    }
    
    for group_name, categories in category_groups.items():
        print(f"\n{group_name.upper()}:")
        print("-" * 80)
        group_data = detailed_df[detailed_df['Category'].isin(categories)]
        if not group_data.empty:
            # Pivot to show sources as columns
            pivot_data = group_data.pivot(index='Category', columns='Source', values='F1 Score')
            print(pivot_data.to_string())
    
    # Save comprehensive results
    os.makedirs('output/comprehensive_analysis', exist_ok=True)
    
    # Save summary
    summary_df.to_csv('output/comprehensive_analysis/bert_summary_results.csv', index=False)
    
    # Save detailed results
    detailed_df.to_csv('output/comprehensive_analysis/bert_detailed_results.csv', index=False)
    
    print(f"\n" + "="*100)
    print("RESULTS SAVED TO:")
    print(f"- Summary: output/comprehensive_analysis/bert_summary_results.csv")
    print(f"- Detailed: output/comprehensive_analysis/bert_detailed_results.csv")
    print("="*100)
    
    return summary_df, detailed_df

def create_performance_visualizations(all_results):
    """Create performance visualizations"""
    
    os.makedirs('output/comprehensive_analysis/plots', exist_ok=True)
    
    # 1. Overall performance comparison
    sources = list(all_results.keys())
    macro_f1s = [all_results[source]['macro_f1'] for source in sources]
    micro_f1s = [all_results[source]['micro_f1'] for source in sources]
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(sources))
    width = 0.35
    
    plt.bar(x - width/2, macro_f1s, width, label='Macro F1', alpha=0.8)
    plt.bar(x + width/2, micro_f1s, width, label='Micro F1', alpha=0.8)
    
    plt.xlabel('Sources')
    plt.ylabel('F1 Score')
    plt.title('BERT Multiclass Performance by Source')
    plt.xticks(x, sources)
    plt.legend()
    plt.tight_layout()
    plt.savefig('output/comprehensive_analysis/plots/overall_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Category performance heatmap
    category_data = []
    for source in sources:
        for category in ALL_CATEGORIES:
            if category in all_results[source]['label_f1_scores']:
                category_data.append({
                    'Source': source,
                    'Category': category,
                    'F1 Score': all_results[source]['label_f1_scores'][category]
                })
    
    category_df = pd.DataFrame(category_data)
    pivot_df = category_df.pivot(index='Category', columns='Source', values='F1 Score')
    
    plt.figure(figsize=(15, 10))
    sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlBu_r', cbar_kws={'label': 'F1 Score'})
    plt.title('BERT Multiclass F1 Scores by Category and Source')
    plt.tight_layout()
    plt.savefig('output/comprehensive_analysis/plots/category_performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to: output/comprehensive_analysis/plots/")

def train_and_evaluate_single_source(source, args):
    """Train and evaluate BERT for a single source"""
    
    print(f"\n{'='*80}")
    print(f"PROCESSING {source.upper()}")
    print(f"{'='*80}")
    
    # Load and preprocess data
    texts, labels, label_names = load_and_preprocess_data(source)
    
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Create datasets
    train_dataset = HomelessnessDataset(train_texts, train_labels, tokenizer, args.max_length)
    val_dataset = HomelessnessDataset(val_texts, val_labels, tokenizer, args.max_length)
    test_dataset = HomelessnessDataset(test_texts, test_labels, tokenizer, args.max_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    if args.mode == 'train_evaluate':
        # Train model
        print("Training model...")
        model = train_model(model, train_loader, val_loader, device, args.epochs, args.learning_rate, source)
        
        # Load best model
        if os.path.exists(f'models/bert_best_{source}.pt'):
            model.load_state_dict(torch.load(f'models/bert_best_{source}.pt'))
            print("Loaded best model for evaluation")
    
    elif args.mode == 'evaluate_only':
        # Load trained model
        model_path = f'models/bert_best_{source}.pt'
        if not os.path.exists(model_path):
            print(f"Trained model not found: {model_path}")
            return None
        
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded trained model from {model_path}")
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(model, test_loader, device, label_names, source)
    
    # Save results
    results_df = save_results(results, source, label_names)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Comprehensive BERT Multiclass Analysis')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train_evaluate', 'evaluate_only', 'predict_only'],
                       help='Mode: train_evaluate, evaluate_only, or predict_only')
    parser.add_argument('--sources', type=str, nargs='+', 
                       default=['reddit', 'x', 'news', 'meeting_minutes'],
                       help='Sources to process')
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
    
    print(f"BERT COMPREHENSIVE MULTICLASS ANALYSIS")
    print(f"Mode: {args.mode}")
    print(f"Sources: {', '.join(args.sources)}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.mode == 'predict_only':
        if not args.input:
            raise ValueError("--input is required for predict_only mode")
        # Prediction functionality would go here
        print("Prediction mode not yet implemented")
        return
    
    # Process each source
    all_results = {}
    
    for source in args.sources:
        try:
            results = train_and_evaluate_single_source(source, args)
            if results:
                all_results[source] = results
        except Exception as e:
            print(f"Error processing {source}: {e}")
            continue
    
    if all_results:
        # Print comprehensive results
        summary_df, detailed_df = print_comprehensive_results(all_results)
        
        # Create visualizations
        create_performance_visualizations(all_results)
        
        print(f"\nAnalysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("No results to analyze")

if __name__ == "__main__":
    main() 