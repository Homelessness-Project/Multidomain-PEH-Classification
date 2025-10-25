#!/usr/bin/env python3
"""
Final BERT Classifier - All improvements combined

This script implements:
1. SMOTE for synthetic data generation
2. Focal Loss for class imbalance
3. Category-specific thresholds
4. Data augmentation
5. Proper data type handling

Usage:
    python scripts/bert_final_classifier.py --source reddit --mode train
"""

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AdamW, get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_recall_fscore_support,
    accuracy_score, hamming_loss, roc_auc_score, average_precision_score
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
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

class FinalDataset(Dataset):
    """Final dataset with augmentation"""
    
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
        
        # Simple augmentation
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
            # Random word dropout (10-20%)
            dropout_rate = np.random.uniform(0.1, 0.2)
            num_drop = max(1, int(len(words) * dropout_rate))
            indices_to_drop = np.random.choice(len(words), num_drop, replace=False)
            words = [word for i, word in enumerate(words) if i not in indices_to_drop]
            
            # Random word insertion
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
            # Aggressive thresholding for rare classes
            if category in ['racist', 'media portrayal']:
                labels[:, i] = (soft_labels_df[category] >= 0.15).astype(int)
            elif category in ['deserving/undeserving', 'not in my backyard', 'ask a genuine question']:
                labels[:, i] = (soft_labels_df[category] >= 0.25).astype(int)
            else:
                labels[:, i] = (soft_labels_df[category] >= 0.4).astype(int)
        else:
            print(f"Warning: Category '{category}' not found in data, using zeros")
    
    # Print detailed statistics
    print(f"\nLabel distribution (before SMOTE):")
    for i, category in enumerate(ALL_CATEGORIES):
        positive_count = np.sum(labels[:, i])
        percentage = positive_count/len(labels)*100
        print(f"  {category}: {positive_count}/{len(labels)} ({percentage:.1f}%)")
    
    return texts, labels, ALL_CATEGORIES

def apply_smote_final(texts, labels, target_ratio=0.3):
    """Apply SMOTE to balance the dataset and track synthetic counts per category"""
    print(f"\nApplying SMOTE augmentation...")
    
    # Convert texts to numerical features for SMOTE
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 2))
    text_features = vectorizer.fit_transform(texts).toarray()
    
    # Apply SMOTE for each category
    augmented_texts = texts.copy()
    augmented_labels = labels.copy()
    synthetic_counts_per_category = {category: 0 for category in ALL_CATEGORIES}
    
    for i, category in enumerate(ALL_CATEGORIES):
        category_labels = labels[:, i]
        positive_ratio = np.mean(category_labels)
        
        if positive_ratio > 0 and positive_ratio < target_ratio:  # Only augment imbalanced categories
            print(f"  Augmenting {category} (current: {positive_ratio:.1%})...")
            
            try:
                # Calculate target number of positive samples
                current_positive = int(np.sum(category_labels))
                target_positive = int(len(category_labels) * target_ratio)
                
                if target_positive > current_positive:
                    # Apply SMOTE
                    smote = SMOTE(sampling_strategy={1: target_positive}, random_state=42)
                    
                    # Combine text features with labels for SMOTE
                    combined_features = np.column_stack([text_features, labels])
                    
                    X_resampled, y_resampled = smote.fit_resample(combined_features, category_labels)
                    
                    # Extract new samples
                    new_text_features = X_resampled[:, :-len(ALL_CATEGORIES)]
                    new_labels = X_resampled[:, -len(ALL_CATEGORIES):]
                    
                    # Find new samples (not in original)
                    original_indices = set(range(len(texts)))
                    new_indices = []
                    
                    for j in range(len(X_resampled)):
                        if j not in original_indices:
                            new_indices.append(j)
                    
                    if len(new_indices) > 0:
                        # Generate synthetic texts (simplified approach)
                        synthetic_texts = []
                        for idx in new_indices:
                            # Find closest original text
                            distances = np.linalg.norm(text_features - new_text_features[idx], axis=1)
                            closest_idx = np.argmin(distances)
                            # Create variation of closest text
                            base_text = texts[closest_idx]
                            synthetic_text = base_text + " " + np.random.choice([
                                "This is important.", "I think this matters.", "This is relevant.",
                                "This is significant.", "This is crucial.", "This is key.",
                                "This is notable.", "This is worth noting.", "This is important to consider."
                            ])
                            synthetic_texts.append(synthetic_text)
                        
                        # Add synthetic samples
                        augmented_texts.extend(synthetic_texts)
                        augmented_labels = np.vstack([augmented_labels, new_labels[new_indices]])
                        synthetic_counts_per_category[category] += len(new_indices)
                        
                        print(f"    Added {len(new_indices)} synthetic samples for {category}")
                    else:
                        print(f"    No new samples generated for {category}")
                
            except Exception as e:
                print(f"    SMOTE failed for {category}: {e}")
                continue
    
    print(f"Dataset size: {len(texts)} -> {len(augmented_texts)}")
    
    # Print new distribution
    print(f"\nLabel distribution (after SMOTE):")
    for i, category in enumerate(ALL_CATEGORIES):
        positive_count = np.sum(augmented_labels[:, i])
        percentage = positive_count/len(augmented_labels)*100
        print(f"  {category}: {positive_count}/{len(augmented_labels)} ({percentage:.1f}%)")
    
    return augmented_texts, augmented_labels, synthetic_counts_per_category

def create_model_and_tokenizer(num_labels, model_name='bert-base-uncased'):
    """Create model and tokenizer.

    Supports 'bert-base-uncased', 'roberta-base', and 'modernbert-base'.
    The latter maps to the HuggingFace repo 'answerdotai/ModernBERT-base'.
    """
    # Map friendly names to HF model identifiers where needed
    if model_name == 'modernbert-base':
        hf_model_name = 'answerdotai/ModernBERT-base'
    else:
        hf_model_name = model_name

    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_model_name, num_labels=num_labels, problem_type='multi_label_classification'
    )

    return model, tokenizer

def train_model(model, train_loader, val_loader, device, epochs=5, learning_rate=2e-5, source='', model_name=''):
    """Train model with SMOTE and focal loss"""
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    
    # Use focal loss
    criterion = FocalLoss(alpha=1, gamma=2)
    
    best_val_f1 = 0
    patience = 5
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
            true_labels = val_true_labels[:, i].astype(int)  # Ensure int type
            pred_scores = val_predictions[:, i]
            
            if len(np.unique(true_labels)) > 1:
                best_f1 = 0
                best_threshold = 0.5
                
                for threshold in np.arange(0.05, 0.95, 0.05):
                    binary_pred = (pred_scores > threshold).astype(int)  # Ensure int type
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
            }, f'models/final_{model_name}_best_{source}.pt')
            print(f"New best {model_name} model saved! Macro F1: {macro_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping {model_name} after {patience} epochs without improvement")
                break
    
    return model

def evaluate_model(model, test_loader, device, label_names, source, model_name=''):
    """Evaluate model with category-specific thresholds"""
    
    # Load best thresholds
    checkpoint_path = f'models/final_{model_name}_best_{source}.pt'
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
    # Additional global metrics
    subset_acc = accuracy_score(final_true_labels, final_predictions)
    ham_loss = hamming_loss(final_true_labels, final_predictions)
    macro_precision = f1_score(final_true_labels, final_predictions, average='macro', zero_division=0, beta=0.0) if False else precision_recall_fscore_support(final_true_labels, final_predictions, average='macro', zero_division=0)[0]
    macro_recall = precision_recall_fscore_support(final_true_labels, final_predictions, average='macro', zero_division=0)[1]
    micro_precision = precision_recall_fscore_support(final_true_labels, final_predictions, average='micro', zero_division=0)[0]
    micro_recall = precision_recall_fscore_support(final_true_labels, final_predictions, average='micro', zero_division=0)[1]
    
    # Per-label F1 scores
    label_f1_scores = {}
    label_precision_scores = {}
    label_recall_scores = {}
    label_accuracy_scores = {}
    label_support_pos = {}
    label_support_neg = {}
    label_roc_auc = {}
    label_average_precision = {}
    
    for i, label_name in enumerate(label_names):
        y_true = final_true_labels[:, i]
        y_pred = final_predictions[:, i]
        y_scores = all_predictions[:, i]

        label_f1 = f1_score(y_true, y_pred, zero_division=0)
        precision, recall, _, support = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )

        # Accuracy per label (TP+TN)/N
        acc = accuracy_score(y_true, y_pred)
        # Support counts
        label_support_pos[label_name] = int((y_true == 1).sum())
        label_support_neg[label_name] = int((y_true == 0).sum())

        # AUC metrics where possible
        try:
            if len(np.unique(y_true)) > 1:
                label_roc_auc[label_name] = float(roc_auc_score(y_true, y_scores))
                label_average_precision[label_name] = float(average_precision_score(y_true, y_scores))
            else:
                label_roc_auc[label_name] = None
                label_average_precision[label_name] = None
        except Exception:
            label_roc_auc[label_name] = None
            label_average_precision[label_name] = None

        label_f1_scores[label_name] = label_f1
        label_precision_scores[label_name] = precision
        label_recall_scores[label_name] = recall
        label_accuracy_scores[label_name] = acc
    
    results = {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'subset_accuracy': subset_acc,
        'hamming_loss': ham_loss,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'best_thresholds': best_thresholds,
        'test_size': len(final_true_labels),
        'num_labels': len(label_names),
        'label_f1_scores': label_f1_scores,
        'label_precision_scores': label_precision_scores,
        'label_recall_scores': label_recall_scores,
        'label_accuracy_scores': label_accuracy_scores,
        'label_support_pos': label_support_pos,
        'label_support_neg': label_support_neg,
        'label_roc_auc': label_roc_auc,
        'label_average_precision': label_average_precision
    }
    
    return results

def save_results(results, source, label_names, model_name='', use_smote=False):
    """Save results to nlp_outputs directory"""
    
    # Create output directory with new structure
    output_dir = f'nlp_outputs/{source}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create model filename
    model_filename = f'{model_name.replace("-", "_")}_smote' if use_smote else f'{model_name.replace("-", "_")}_original'
    
    # Save metrics
    metrics = {
        'source': source,
        'model': f'{model_name}-final',
        'use_smote': use_smote,
        'macro_f1': results['macro_f1'],
        'micro_f1': results['micro_f1'],
        'subset_accuracy': results.get('subset_accuracy'),
        'hamming_loss': results.get('hamming_loss'),
        'macro_precision': results.get('macro_precision'),
        'macro_recall': results.get('macro_recall'),
        'micro_precision': results.get('micro_precision'),
        'micro_recall': results.get('micro_recall'),
        'best_thresholds': results['best_thresholds'],
        'train_size': results.get('train_size'),
        'val_size': results.get('val_size'),
        'test_size': results.get('test_size'),
        'num_labels': results['num_labels'],
        'label_f1_scores': results['label_f1_scores'],
        'label_precision_scores': results['label_precision_scores'],
        'label_recall_scores': results['label_recall_scores'],
        'label_accuracy_scores': results.get('label_accuracy_scores'),
        'label_support_pos': results.get('label_support_pos'),
        'label_support_neg': results.get('label_support_neg'),
        'label_roc_auc': results.get('label_roc_auc'),
        'label_average_precision': results.get('label_average_precision'),
        'synthetic_counts_per_category': results.get('synthetic_counts_per_category'),
        'total_synthetic_samples': results.get('total_synthetic_samples')
    }
    
    with open(f'{output_dir}/{model_filename}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save detailed results
    results_data = []
    for i, label in enumerate(label_names):
        results_data.append({
            'category': label,
            'f1_score': results['label_f1_scores'][label],
            'precision': results['label_precision_scores'][label],
            'recall': results['label_recall_scores'][label],
            'threshold': results['best_thresholds'][i],
            'synthetic_added': (results.get('synthetic_counts_per_category') or {}).get(label, 0)
        })
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(f'{output_dir}/{model_filename}.csv', index=False)
    
    print(f"\nResults saved to {output_dir}/{model_filename}.csv")
    print(f"Macro F1: {results['macro_f1']:.4f}")
    print(f"Micro F1: {results['micro_f1']:.4f}")
    
    # Print per-category results
    print(f"\nPer-category F1 scores ({model_name}):")
    for i, label in enumerate(label_names):
        f1 = results['label_f1_scores'][label]
        threshold = results['best_thresholds'][i]
        print(f"  {label}: {f1:.4f} (threshold: {threshold:.3f})")
    
    # Append per-category summary row(s) to a global CSV: nlp_outputs/all_transformer_results.csv
    # Columns: category, country, train, val, test, synthetic
    try:
        summary_rows = []
        for i, label in enumerate(label_names):
            summary_rows.append({
                'category': label,
                'source': source,
                'model': model_name,
                'smote': bool(use_smote),
                'train': results.get('train_size'),
                'val': results.get('val_size'),
                'test': results.get('test_size'),
                'synthetic': (results.get('synthetic_counts_per_category') or {}).get(label, 0),
                'macro_f1': results['macro_f1'],
                'micro_f1': results['micro_f1'],
                'subset_accuracy': results.get('subset_accuracy'),
                'hamming_loss': results.get('hamming_loss'),
                'precision': results['label_precision_scores'][label],
                'recall': results['label_recall_scores'][label],
                'f1': results['label_f1_scores'][label],
                'accuracy': (results.get('label_accuracy_scores') or {}).get(label),
                'roc_auc': (results.get('label_roc_auc') or {}).get(label),
                'average_precision': (results.get('label_average_precision') or {}).get(label)
            })
        summary_df = pd.DataFrame(summary_rows)
        master_csv_path = 'nlp_outputs/all_transformer_results.csv'
        os.makedirs('nlp_outputs', exist_ok=True)
        write_header = not os.path.exists(master_csv_path)
        # Upgrade existing master CSV schema if needed (rename country->source, add model/smote columns)
        if not write_header:
            try:
                with open(master_csv_path, 'r') as f:
                    header_line = f.readline().strip()
                if 'model' not in header_line or 'source' not in header_line:
                    df_existing = pd.read_csv(master_csv_path)
                    if 'country' in df_existing.columns and 'source' not in df_existing.columns:
                        df_existing = df_existing.rename(columns={'country': 'source'})
                    if 'model' not in df_existing.columns:
                        df_existing['model'] = None
                    if 'smote' not in df_existing.columns:
                        df_existing['smote'] = None
                    # Move columns to a consistent order if possible
                    preferred = ['category','source','model','smote','train','val','test','synthetic','macro_f1','micro_f1','subset_accuracy','hamming_loss','precision','recall','f1','accuracy','roc_auc','average_precision']
                    cols = [c for c in preferred if c in df_existing.columns] + [c for c in df_existing.columns if c not in preferred]
                    df_existing = df_existing[cols]
                    df_existing.to_csv(master_csv_path, index=False)
            except Exception as exc:
                print(f"Warning: failed to upgrade all_transformer_results.csv schema: {exc}")
        summary_df.to_csv(master_csv_path, mode='a', header=write_header, index=False)
        print(f"Appended per-category summary to {master_csv_path}")

        # Append overall summary row to nlp_outputs/summary_overall_results.csv
        try:
            overall_row = [{
                'source': source,
                'model': model_name,
                'smote': bool(use_smote),
                'macro_f1': results['macro_f1'],
                'micro_f1': results['micro_f1'],
                'subset_accuracy': results.get('subset_accuracy'),
                'hamming_loss': results.get('hamming_loss'),
                'train': results.get('train_size'),
                'val': results.get('val_size'),
                'test': results.get('test_size'),
                'total_synthetic': results.get('total_synthetic_samples')
            }]
            overall_df = pd.DataFrame(overall_row)
            overall_path = 'nlp_outputs/summary_overall_results.csv'
            write_header_overall = not os.path.exists(overall_path)
            overall_df.to_csv(overall_path, mode='a', header=write_header_overall, index=False)
            print(f"Appended overall summary to {overall_path}")
        except Exception as exc:
            print(f"Warning: failed to append overall summary CSV: {exc}")

        # Also append to unified all_results.csv (transformers + traditional)
        unified_csv_path = 'nlp_outputs/all_results.csv'
        write_header_unified = not os.path.exists(unified_csv_path)
        summary_df.to_csv(unified_csv_path, mode='a', header=write_header_unified, index=False)
        print(f"Appended per-category summary to {unified_csv_path}")
    except Exception as exc:
        print(f"Warning: failed to append to global results CSV: {exc}")

    return results_df

def main():
    parser = argparse.ArgumentParser(description='Final BERT Classifier')
    parser.add_argument('--source', type=str, required=True, 
                       choices=['reddit', 'x', 'news', 'meeting_minutes'],
                       help='Data source to use')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'evaluate'],
                       help='Mode: train or evaluate')
    parser.add_argument('--model', type=str, default='bert-base-uncased',
                       choices=['bert-base-uncased', 'roberta-base', 'modernbert-base'],
                       help='Model to use')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=256, help='Max sequence length')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--val_size', type=float, default=0.1, help='Validation set size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use_smote', action='store_true', help='Use SMOTE for data augmentation')
    
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
        
        # Split data FIRST (before SMOTE to prevent data leakage)
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels, test_size=args.test_size + args.val_size, random_state=args.seed
        )
        
        val_size_ratio = args.val_size / (args.test_size + args.val_size)
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels, test_size=1-val_size_ratio, random_state=args.seed
        )
        
        # Apply SMOTE ONLY to training data if requested
        synthetic_counts_per_category = None
        total_synthetic_samples = None
        if args.use_smote:
            train_texts, train_labels, synthetic_counts_per_category = apply_smote_final(train_texts, train_labels)
        
        print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
        
        # Create model and tokenizer
        model, tokenizer = create_model_and_tokenizer(len(label_names), args.model)
        model.to(device)
        
        # Create datasets
        train_dataset = FinalDataset(train_texts, train_labels, tokenizer, args.max_length, augment=True)
        val_dataset = FinalDataset(val_texts, val_labels, tokenizer, args.max_length)
        test_dataset = FinalDataset(test_texts, test_labels, tokenizer, args.max_length)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Train model
        print(f"Training {args.model} with final approach...")
        model = train_model(model, train_loader, val_loader, device, args.epochs, args.learning_rate, args.source, args.model)
        
        # Load best model
        checkpoint_path = f'models/final_{args.model}_best_{args.source}.pt'
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded best model for evaluation")
        
        # Evaluate model
        print("Evaluating model...")
        results = evaluate_model(model, test_loader, device, label_names, args.source, args.model)
        # Attach split sizes and synthetic counts
        results['train_size'] = len(train_texts)
        results['val_size'] = len(val_texts)
        results['test_size'] = len(test_texts)
        if synthetic_counts_per_category is not None:
            results['synthetic_counts_per_category'] = synthetic_counts_per_category
            results['total_synthetic_samples'] = int(sum(synthetic_counts_per_category.values()))
        
        # Save results
        save_results(results, args.source, label_names, args.model, args.use_smote)
        
    elif args.mode == 'evaluate':
        # Load and preprocess data
        texts, labels, label_names = load_and_preprocess_data(args.source)
        
        # Apply SMOTE if requested
        synthetic_counts_per_category = None
        total_synthetic_samples = None
        if args.use_smote:
            texts, labels, synthetic_counts_per_category = apply_smote_final(texts, labels)
        
        # Split data
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=args.test_size, random_state=args.seed
        )
        
        # Load model
        model, tokenizer = create_model_and_tokenizer(len(label_names), args.model)
        model.to(device)
        
        # Load trained weights
        checkpoint_path = f'models/final_{args.model}_best_{args.source}.pt'
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Trained model not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded trained model from {checkpoint_path}")
        
        # Create test dataset
        test_dataset = FinalDataset(test_texts, test_labels, tokenizer, args.max_length)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Evaluate
        print("Evaluating model...")
        results = evaluate_model(model, test_loader, device, label_names, args.source, args.model)
        # Attach split sizes and synthetic counts
        results['train_size'] = len(train_texts)
        results['val_size'] = 0  # no val split in evaluate mode
        results['test_size'] = len(test_texts)
        if synthetic_counts_per_category is not None:
            results['synthetic_counts_per_category'] = synthetic_counts_per_category
            results['total_synthetic_samples'] = int(sum(synthetic_counts_per_category.values()))
        
        # Save results
        save_results(results, args.source, label_names, args.model, args.use_smote)

if __name__ == "__main__":
    main()
