#!/usr/bin/env python3
"""
Working Benchmark: BERT, RoBERTa, Linear Regression, and SVM

This script compares:
1. BERT (with SMOTE)
2. RoBERTa (with SMOTE) 
3. Logistic Regression (with SMOTE) - One-vs-Rest
4. SVM (with SMOTE) - One-vs-Rest

Usage:
    python scripts/working_benchmark.py --source reddit --mode train
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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
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
        inputs = torch.sigmoid(inputs)
        bce_loss = -targets * torch.log(inputs + 1e-8) - (1 - targets) * torch.log(1 - inputs + 1e-8)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class TransformerDataset(Dataset):
    """Dataset for transformer models"""
    
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
        words = text.split()
        if len(words) > 3:
            dropout_rate = np.random.uniform(0.1, 0.2)
            num_drop = max(1, int(len(words) * dropout_rate))
            indices_to_drop = np.random.choice(len(words), num_drop, replace=False)
            words = [word for i, word in enumerate(words) if i not in indices_to_drop]
            
            if len(words) > 1 and np.random.random() < 0.1:
                insert_pos = np.random.randint(0, len(words))
                words.insert(insert_pos, np.random.choice(['very', 'really', 'quite', 'rather']))
            
            text = ' '.join(words)
        return text

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
    
    # Create labels matrix with aggressive thresholding
    labels = np.zeros((len(soft_labels_df), len(ALL_CATEGORIES)))
    
    for i, category in enumerate(ALL_CATEGORIES):
        if category in soft_labels_df.columns:
            if category in ['racist', 'media portrayal']:
                labels[:, i] = (soft_labels_df[category] >= 0.15).astype(int)
            elif category in ['deserving/undeserving', 'not in my backyard', 'ask a genuine question']:
                labels[:, i] = (soft_labels_df[category] >= 0.25).astype(int)
            else:
                labels[:, i] = (soft_labels_df[category] >= 0.4).astype(int)
        else:
            print(f"Warning: Category '{category}' not found in data, using zeros")
    
    # Print detailed statistics
    print(f"\nLabel distribution:")
    for i, category in enumerate(ALL_CATEGORIES):
        positive_count = np.sum(labels[:, i])
        percentage = positive_count/len(labels)*100
        print(f"  {category}: {positive_count}/{len(labels)} ({percentage:.1f}%)")
    
    return texts, labels, ALL_CATEGORIES

def apply_smote_simple(texts, labels, target_ratio=0.3):
    """Apply SMOTE to balance the dataset"""
    print(f"\nApplying SMOTE augmentation...")
    
    # Convert texts to numerical features for SMOTE
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
    text_features = vectorizer.fit_transform(texts).toarray()
    
    # Apply SMOTE for each category
    augmented_texts = texts.copy()
    augmented_labels = labels.copy()
    
    for i, category in enumerate(ALL_CATEGORIES):
        category_labels = labels[:, i]
        positive_ratio = np.mean(category_labels)
        
        if positive_ratio > 0 and positive_ratio < target_ratio:
            print(f"  Augmenting {category} (current: {positive_ratio:.1%})...")
            
            try:
                current_positive = int(np.sum(category_labels))
                target_positive = int(len(category_labels) * target_ratio)
                
                if target_positive > current_positive:
                    smote = SMOTE(sampling_strategy={1: target_positive}, random_state=42)
                    combined_features = np.column_stack([text_features, labels])
                    
                    X_resampled, y_resampled = smote.fit_resample(combined_features, category_labels)
                    
                    new_text_features = X_resampled[:, :-len(ALL_CATEGORIES)]
                    new_labels = X_resampled[:, -len(ALL_CATEGORIES):]
                    
                    original_indices = set(range(len(texts)))
                    new_indices = []
                    
                    for j in range(len(X_resampled)):
                        if j not in original_indices:
                            new_indices.append(j)
                    
                    if len(new_indices) > 0:
                        synthetic_texts = []
                        for idx in new_indices:
                            distances = np.linalg.norm(text_features - new_text_features[idx], axis=1)
                            closest_idx = np.argmin(distances)
                            base_text = texts[closest_idx]
                            synthetic_text = base_text + " " + np.random.choice([
                                "This is important.", "I think this matters.", "This is relevant.",
                                "This is significant.", "This is crucial.", "This is key."
                            ])
                            synthetic_texts.append(synthetic_text)
                        
                        augmented_texts.extend(synthetic_texts)
                        augmented_labels = np.vstack([augmented_labels, new_labels[new_indices]])
                        
                        print(f"    Added {len(new_indices)} synthetic samples for {category}")
                
            except Exception as e:
                print(f"    SMOTE failed for {category}: {e}")
                continue
    
    print(f"Dataset size: {len(texts)} -> {len(augmented_texts)}")
    return augmented_texts, augmented_labels

def train_transformer_model(model, tokenizer, train_loader, val_loader, device, epochs=3, learning_rate=2e-5, model_name=''):
    """Train transformer model"""
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    
    criterion = FocalLoss(alpha=1, gamma=2)
    
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
            true_labels = val_true_labels[:, i].astype(int)
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
        
        # Calculate micro F1
        global_threshold = 0.5
        binary_pred_global = (val_predictions > global_threshold).astype(int)
        binary_true_global = (val_true_labels > 0.5).astype(int)
        micro_f1 = f1_score(binary_true_global, binary_pred_global, average='micro', zero_division=0)
        
        print(f"{model_name} Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
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
            }, f'models/benchmark_{model_name}_best.pt')
            print(f"New best {model_name} model saved! Macro F1: {macro_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping {model_name} after {patience} epochs without improvement")
                break
    
    return model

def train_sklearn_models(X_train, y_train, X_test, y_test):
    """Train sklearn models with One-vs-Rest"""
    
    results = {}
    
    # 1. Logistic Regression
    print("\nTraining Logistic Regression (One-vs-Rest)...")
    lr_model = MultiOutputClassifier(
        LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    )
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_f1_macro = f1_score(y_test, lr_pred, average='macro', zero_division=0)
    lr_f1_micro = f1_score(y_test, lr_pred, average='micro', zero_division=0)
    
    results['logistic_regression'] = {
        'macro_f1': lr_f1_macro,
        'micro_f1': lr_f1_micro,
        'predictions': lr_pred
    }
    
    print(f"Logistic Regression - Macro F1: {lr_f1_macro:.4f}, Micro F1: {lr_f1_micro:.4f}")
    
    # 2. SVM
    print("\nTraining SVM (One-vs-Rest)...")
    svm_model = MultiOutputClassifier(
        SVC(random_state=42, class_weight='balanced', probability=True)
    )
    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)
    svm_f1_macro = f1_score(y_test, svm_pred, average='macro', zero_division=0)
    svm_f1_micro = f1_score(y_test, svm_pred, average='micro', zero_division=0)
    
    results['svm'] = {
        'macro_f1': svm_f1_macro,
        'micro_f1': svm_f1_micro,
        'predictions': svm_pred
    }
    
    print(f"SVM - Macro F1: {svm_f1_macro:.4f}, Micro F1: {svm_f1_micro:.4f}")
    
    # 3. Random Forest
    print("\nTraining Random Forest (One-vs-Rest)...")
    rf_model = MultiOutputClassifier(
        RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100)
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_f1_macro = f1_score(y_test, rf_pred, average='macro', zero_division=0)
    rf_f1_micro = f1_score(y_test, rf_pred, average='micro', zero_division=0)
    
    results['random_forest'] = {
        'macro_f1': rf_f1_macro,
        'micro_f1': rf_f1_micro,
        'predictions': rf_pred
    }
    
    print(f"Random Forest - Macro F1: {rf_f1_macro:.4f}, Micro F1: {rf_f1_micro:.4f}")
    
    return results

def evaluate_transformer_model(model, test_loader, device, label_names, model_name=''):
    """Evaluate transformer model"""
    
    # Load best thresholds
    checkpoint_path = f'models/benchmark_{model_name}_best.pt'
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
        for batch in tqdm(test_loader, desc=f"Evaluating {model_name}"):
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
    for i, label_name in enumerate(label_names):
        label_f1 = f1_score(
            final_true_labels[:, i], final_predictions[:, i], zero_division=0
        )
        label_f1_scores[label_name] = label_f1
    
    results = {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'best_thresholds': best_thresholds,
        'label_f1_scores': label_f1_scores,
        'predictions': final_predictions
    }
    
    return results

def save_benchmark_results(all_results, source, label_names):
    """Save benchmark results"""
    
    output_dir = f'output/{source}/benchmark'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create comparison table
    comparison_data = []
    for model_name, results in all_results.items():
        comparison_data.append({
            'model': model_name,
            'macro_f1': results['macro_f1'],
            'micro_f1': results['micro_f1']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('macro_f1', ascending=False)
    
    # Save comparison
    comparison_df.to_csv(f'{output_dir}/benchmark_comparison_{source}.csv', index=False)
    
    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS FOR {source.upper()}")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Macro F1':<10} {'Micro F1':<10}")
    print(f"{'-'*60}")
    for _, row in comparison_df.iterrows():
        print(f"{row['model']:<20} {row['macro_f1']:<10.4f} {row['micro_f1']:<10.4f}")
    
    # Save detailed results
    for model_name, results in all_results.items():
        if 'label_f1_scores' in results:
            results_data = []
            for i, label in enumerate(label_names):
                results_data.append({
                    'category': label,
                    'f1_score': results['label_f1_scores'][label],
                    'threshold': results.get('best_thresholds', [0.5] * len(label_names))[i]
                })
            
            results_df = pd.DataFrame(results_data)
            results_df.to_csv(f'{output_dir}/benchmark_{model_name}_detailed_{source}.csv', index=False)
    
    # Save summary
    summary = {
        'source': source,
        'models_tested': list(all_results.keys()),
        'best_model': comparison_df.iloc[0]['model'],
        'best_macro_f1': comparison_df.iloc[0]['macro_f1'],
        'best_micro_f1': comparison_df.iloc[0]['micro_f1'],
        'comparison': comparison_df.to_dict('records')
    }
    
    with open(f'{output_dir}/benchmark_summary_{source}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/")
    print(f"Best model: {comparison_df.iloc[0]['model']} (Macro F1: {comparison_df.iloc[0]['macro_f1']:.4f})")
    
    return comparison_df

def main():
    parser = argparse.ArgumentParser(description='Working Benchmark Classifiers')
    parser.add_argument('--source', type=str, required=True, 
                       choices=['reddit', 'x', 'news', 'meeting_minutes'],
                       help='Data source to use')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'evaluate'],
                       help='Mode: train or evaluate')
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs for transformers')
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
        
        # Prepare data for sklearn models
        print("\nPreparing data for sklearn models...")
        sklearn_texts, sklearn_labels = apply_smote_simple(train_texts, train_labels)
        
        # Convert texts to TF-IDF features
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        X_train = vectorizer.fit_transform(sklearn_texts).toarray()
        X_test = vectorizer.transform(test_texts).toarray()
        
        # Train sklearn models
        sklearn_results = train_sklearn_models(X_train, sklearn_labels, X_test, test_labels)
        
        # Train transformer models
        all_results = {}
        
        # BERT
        print("\nTraining BERT...")
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_labels=len(label_names), problem_type='multi_label_classification'
        )
        bert_model.to(device)
        
        # Apply SMOTE for BERT
        bert_texts, bert_labels = apply_smote_simple(train_texts, train_labels)
        
        # Create datasets
        train_dataset = TransformerDataset(bert_texts, bert_labels, bert_tokenizer, args.max_length, augment=True)
        val_dataset = TransformerDataset(val_texts, val_labels, bert_tokenizer, args.max_length)
        test_dataset = TransformerDataset(test_texts, test_labels, bert_tokenizer, args.max_length)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Train BERT
        train_transformer_model(bert_model, bert_tokenizer, train_loader, val_loader, device, args.epochs, args.learning_rate, 'bert')
        
        # Evaluate BERT
        bert_results = evaluate_transformer_model(bert_model, test_loader, device, label_names, 'bert')
        all_results['bert'] = bert_results
        
        # RoBERTa
        print("\nTraining RoBERTa...")
        roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        roberta_model = RobertaForSequenceClassification.from_pretrained(
            'roberta-base', num_labels=len(label_names), problem_type='multi_label_classification'
        )
        roberta_model.to(device)
        
        # Create datasets
        train_dataset = TransformerDataset(bert_texts, bert_labels, roberta_tokenizer, args.max_length, augment=True)
        val_dataset = TransformerDataset(val_texts, val_labels, roberta_tokenizer, args.max_length)
        test_dataset = TransformerDataset(test_texts, test_labels, roberta_tokenizer, args.max_length)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Train RoBERTa
        train_transformer_model(roberta_model, roberta_tokenizer, train_loader, val_loader, device, args.epochs, args.learning_rate, 'roberta')
        
        # Evaluate RoBERTa
        roberta_results = evaluate_transformer_model(roberta_model, test_loader, device, label_names, 'roberta')
        all_results['roberta'] = roberta_results
        
        # Add sklearn results
        all_results.update(sklearn_results)
        
        # Save results
        save_benchmark_results(all_results, args.source, label_names)
        
    elif args.mode == 'evaluate':
        print("Evaluation mode not implemented yet. Use train mode to run the benchmark.")
        return

if __name__ == "__main__":
    main()

