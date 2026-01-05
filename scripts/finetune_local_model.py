#!/usr/bin/env python3
"""
Fine-tuning script for local models like Qwen

This script supports fine-tuning transformer models (e.g., Qwen, Llama, etc.)
for multi-label classification on homelessness-related content.

Based on classify_comments.py structure with fine-tuning capabilities.

Usage:
    # Fine-tune Qwen model (zero-shot prompts)
    python scripts/finetune_local_model.py \
        --source reddit \
        --model qwen \
        --mode train \
        --epochs 3 \
        --use_smote \
        --few_shot none

    # Fine-tune with few-shot prompts
    python scripts/finetune_local_model.py \
        --source reddit \
        --model qwen \
        --mode train \
        --epochs 3 \
        --use_smote \
        --few_shot reddit

    # Evaluate trained model
    python scripts/finetune_local_model.py \
        --source reddit \
        --model qwen \
        --mode evaluate \
        --few_shot none
"""

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoModel,
    AutoModelForCausalLM, AdamW, get_linear_schedule_with_warmup,
    AutoConfig
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_recall_fscore_support
)
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import os
import sys
import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import utils - handle both direct import and script import
try:
    from scripts.utils import (
        get_model_config,
        create_classification_prompt
    )
except ImportError:
    # Fallback for when running as script
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scripts.utils import (
        get_model_config,
        create_classification_prompt
    )

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

class FineTuningDataset(Dataset):
    """Dataset for fine-tuning with optional prompt-based formatting"""
    
    def __init__(self, texts, labels, tokenizer, max_length=256, augment=False, 
                 use_prompt=False, source='reddit', few_shot='none'):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        self.use_prompt = use_prompt
        self.source = source
        self.few_shot = few_shot
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Simple augmentation
        if self.augment and np.random.random() < 0.3:
            text = self._augment_text(text)
        
        # Apply prompt formatting if requested (for zero-shot or few-shot fine-tuning)
        if self.use_prompt:
            text = create_classification_prompt(
                text, 
                content_type=self.source, 
                few_shot_text=self.few_shot
            )
        
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
    """Load and preprocess data - similar to classify_comments.py structure"""
    print(f"Loading data for {source}...")
    
    # Load soft labels
    soft_labels_path = f'output/annotation/soft_labels/{source}_soft_labels.csv'
    if not os.path.exists(soft_labels_path):
        raise FileNotFoundError(f"Soft labels file not found: {soft_labels_path}")
    
    soft_labels_df = pd.read_csv(soft_labels_path)
    print(f"Loaded {len(soft_labels_df)} soft label samples")
    
    # Load text data - use gold_standard like classify_comments.py
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
    
    # Get text column - similar to classify_comments.py column mapping
    text_col_map = {
        'reddit': 'Comment',
        'x': 'Deidentified_text',
        'news': 'Deidentified_paragraph_text',
        'meeting_minutes': 'Deidentified_paragraph'
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
    
    # Create labels matrix with 0.5 threshold for all categories
    labels = np.zeros((len(soft_labels_df), len(ALL_CATEGORIES)))
    
    for i, category in enumerate(ALL_CATEGORIES):
        if category in soft_labels_df.columns:
            # Use 0.5 threshold for all categories
            labels[:, i] = (soft_labels_df[category] >= 0.5).astype(int)
        else:
            print(f"Warning: Category '{category}' not found in data, using zeros")
    
    # Print detailed statistics
    print(f"\nLabel distribution (before SMOTE):")
    for i, category in enumerate(ALL_CATEGORIES):
        positive_count = np.sum(labels[:, i])
        percentage = positive_count/len(labels)*100
        print(f"  {category}: {positive_count}/{len(labels)} ({percentage:.1f}%)")
    
    return texts, labels, ALL_CATEGORIES

def apply_smote(texts, labels, target_ratio=0.3):
    """Apply SMOTE to balance the dataset"""
    print(f"\nApplying SMOTE augmentation...")
    
    # Convert texts to numerical features for SMOTE
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1, 2))
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
    
    return augmented_texts, augmented_labels

def create_model_and_tokenizer(num_labels, model_name, trust_remote_code=True, device=None):
    """Create model and tokenizer using get_model_config like classify_comments.py"""
    print(f"Loading model: {model_name}")
    
    # Get model config from utils (like classify_comments.py)
    model_config = get_model_config(model_name)
    model_id = model_config["model_id"]
    
    # Determine torch dtype based on device
    if device and device.type == 'mps':
        # MPS works best with float32, but we can try float16 for memory savings
        torch_dtype = torch.float16
        print("Using float16 for MPS to save memory")
    elif device and device.type == 'cuda':
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code
        )
        
        # Check if tokenizer has pad_token, if not set it
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Try to load as sequence classification model
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_id,
                num_labels=num_labels,
                problem_type='multi_label_classification',
                trust_remote_code=trust_remote_code,
                torch_dtype=torch_dtype
            )
        except Exception as e:
            print(f"Warning: Could not load as sequence classification model: {e}")
            print("Attempting to load base model and add classification head...")
            
            # Try AutoModelForCausalLM first (for models like Phi-4, Qwen, Llama)
            try:
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=trust_remote_code,
                    torch_dtype=torch_dtype
                )
            except Exception as e2:
                print(f"Warning: Could not load as causal LM: {e2}")
                print("Trying AutoModel as fallback...")
                # Fallback to AutoModel
                base_model = AutoModel.from_pretrained(
                    model_id,
                    trust_remote_code=trust_remote_code,
                    torch_dtype=torch_dtype
                )
            
            # Create a wrapper model with classification head
            class ModelWithClassificationHead(nn.Module):
                def __init__(self, base_model, num_labels):
                    super().__init__()
                    self.base_model = base_model
                    hidden_size = base_model.config.hidden_size
                    self.classifier = nn.Linear(hidden_size, num_labels)
                    
                def forward(self, input_ids=None, attention_mask=None, labels=None):
                    outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
                    
                    # Handle different model types
                    if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                        # BERT-style models with pooler
                        pooled = outputs.pooler_output
                    elif hasattr(outputs, 'last_hidden_state'):
                        # For causal models (Phi-4, Qwen, Llama), use the last token of each sequence
                        last_hidden_state = outputs.last_hidden_state
                        if attention_mask is not None:
                            # Get the last non-padding token for each sequence
                            seq_lengths = attention_mask.sum(dim=1) - 1
                            batch_size = last_hidden_state.shape[0]
                            pooled = last_hidden_state[torch.arange(batch_size), seq_lengths]
                        else:
                            # Fallback to last token
                            pooled = last_hidden_state[:, -1, :]
                    else:
                        # Fallback to mean pooling if available
                        pooled = outputs.last_hidden_state.mean(dim=1)
                    
                    logits = self.classifier(pooled)
                    return type('Output', (), {'logits': logits})()
            
            model = ModelWithClassificationHead(base_model, num_labels)
        
        # Resize token embeddings if needed
        if len(tokenizer) != model.config.vocab_size:
            if hasattr(model, 'resize_token_embeddings'):
                model.resize_token_embeddings(len(tokenizer))
            elif hasattr(model, 'base_model') and hasattr(model.base_model, 'resize_token_embeddings'):
                model.base_model.resize_token_embeddings(len(tokenizer))
        
        print(f"Model loaded successfully!")
        print(f"  Model type: {type(model).__name__}")
        print(f"  Vocabulary size: {len(tokenizer)}")
        print(f"  Number of labels: {num_labels}")
        
        return model, tokenizer
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_name} ({model_id}): {e}")

def train_model(model, train_loader, val_loader, device, epochs=5, learning_rate=2e-5, 
                source='', model_name='', use_focal_loss=True):
    """Train model with optional SMOTE and focal loss"""
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    
    # Use focal loss if requested, otherwise BCE
    if use_focal_loss:
        criterion = FocalLoss(alpha=1, gamma=2)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
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
            
            # Handle different output formats
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            loss = criterion(logits, labels)
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
                
                # Handle different output formats
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                predictions = torch.sigmoid(logits)
                
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
            model_save_name = model_name.replace('/', '_').replace('\\', '_')
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_thresholds': best_thresholds,
                'macro_f1': macro_f1,
                'micro_f1': micro_f1,
                'model_config': model.config if hasattr(model, 'config') else None
            }, f'models/final_local_{model_save_name}_best_{source}.pt')
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
    model_save_name = model_name.replace('/', '_').replace('\\', '_')
    checkpoint_path = f'models/final_local_{model_save_name}_best_{source}.pt'
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
            
            # Handle different output formats
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            predictions = torch.sigmoid(logits)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    
    # Apply category-specific thresholds
    final_predictions = np.zeros_like(all_predictions)
    for i, threshold in enumerate(best_thresholds):
        final_predictions[:, i] = (all_predictions[:, i] > threshold).astype(int)
    
    binary_true_labels = (all_true_labels > 0.5).astype(int)
    
    # Calculate metrics
    macro_f1 = f1_score(binary_true_labels, final_predictions, average='macro', zero_division=0)
    micro_f1 = f1_score(binary_true_labels, final_predictions, average='micro', zero_division=0)
    
    # Per-category metrics
    category_metrics = {}
    for i, category in enumerate(label_names):
        precision, recall, f1, support = precision_recall_fscore_support(
            binary_true_labels[:, i], final_predictions[:, i], 
            average='binary', zero_division=0
        )
        category_metrics[category] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        }
    
    results = {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'best_thresholds': best_thresholds,
        'category_metrics': category_metrics,
        'test_size': len(all_true_labels)
    }
    
    return results

def save_results(results, source, label_names, model_name='', few_shot='none'):
    """Save results to output directory - similar structure to classify_comments.py"""
    
    model_save_name = model_name.replace('/', '_').replace('\\', '_')
    output_dir = f'output/{source}/finetuned_{model_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics JSON
    metrics = {
        'source': source,
        'model': model_name,
        'few_shot': few_shot,
        'macro_f1': results['macro_f1'],
        'micro_f1': results['micro_f1'],
        'best_thresholds': results['best_thresholds'],
        'test_size': results['test_size'],
        'num_labels': len(label_names),
        'category_metrics': results['category_metrics']
    }
    
    metrics_file = f'{output_dir}/finetuned_{model_save_name}_{source}_{few_shot}_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save detailed CSV
    results_data = []
    for category in label_names:
        if category in results['category_metrics']:
            metrics_dict = results['category_metrics'][category]
            results_data.append({
                'category': category,
                'precision': metrics_dict['precision'],
                'recall': metrics_dict['recall'],
                'f1_score': metrics_dict['f1'],
                'support': metrics_dict['support'],
                'threshold': results['best_thresholds'][label_names.index(category)]
            })
    
    results_df = pd.DataFrame(results_data)
    csv_file = f'{output_dir}/finetuned_{model_save_name}_{source}_{few_shot}_results.csv'
    results_df.to_csv(csv_file, index=False)
    
    print(f"\nResults saved:")
    print(f"  Metrics: {metrics_file}")
    print(f"  CSV: {csv_file}")
    print(f"\nOverall Performance:")
    print(f"  Macro F1: {results['macro_f1']:.4f}")
    print(f"  Micro F1: {results['micro_f1']:.4f}")

def main():
    print("Starting main() function...")
    parser = argparse.ArgumentParser(description="Fine-tune models for homelessness classification")
    parser.add_argument('--model', type=str, default='qwen', 
                       choices=['llama', 'qwen', 'gemma3', 'phi4'],
                       help='Model to use (llama, qwen, gemma3, or phi4)')
    parser.add_argument('--source', type=str, required=True,
                       choices=['reddit', 'x', 'news', 'meeting_minutes'],
                       help='Specify the data source (required)')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'evaluate'],
                       help='Mode: train or evaluate')
    parser.add_argument('--few_shot', type=str, default='none',
                       choices=['reddit', 'x', 'news', 'meeting_minutes', 'none'],
                       help='Few-shot examples to include in prompts (use "none" for zero-shot). Only used if --use_prompt is set.')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size (default: 4 for M3 Mac, increase if you have more memory)')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--max_length', type=int, default=256,
                       help='Max sequence length')
    parser.add_argument('--use_smote', action='store_true',
                       help='Use SMOTE data augmentation')
    parser.add_argument('--use_focal_loss', action='store_true', default=True,
                       help='Use Focal Loss (default: True)')
    parser.add_argument('--use_prompt', action='store_true',
                       help='Use prompt-based fine-tuning (zero-shot or few-shot format)')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size')
    parser.add_argument('--val_size', type=float, default=0.1,
                       help='Validation set size')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Setup logging if running with nohup (like classify_comments.py)
    if not sys.stdout.isatty():
        os.makedirs('logs', exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"logs/finetune_{args.source}_{args.model}_{args.few_shot}_{timestamp}.log"
        sys.stdout = open(log_filename, 'w')
        sys.stderr = sys.stdout
        print(f"Logging to: {log_filename}")
        print(f"Started at: {datetime.datetime.now()}")
        print(f"Command: {' '.join(sys.argv)}")
        print("-" * 80)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device - prioritize MPS for Apple Silicon
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess data
    texts, labels, label_names = load_and_preprocess_data(args.source)
    
    # Apply SMOTE if requested
    if args.use_smote:
        texts, labels = apply_smote(texts, labels)
    
    # Split data
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=args.test_size + args.val_size, random_state=args.seed
    )
    
    val_size_ratio = args.val_size / (args.test_size + args.val_size)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=1-val_size_ratio, random_state=args.seed
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(train_texts)}")
    print(f"  Val: {len(val_texts)}")
    print(f"  Test: {len(test_texts)}")
    
    # Create model and tokenizer (using get_model_config like classify_comments.py)
    model, tokenizer = create_model_and_tokenizer(
        num_labels=len(label_names),
        model_name=args.model,
        trust_remote_code=True,
        device=device
    )
    
    model.to(device)
    
    # Enable gradient checkpointing for memory efficiency (M3 Mac optimization)
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled for memory efficiency")
    elif hasattr(model, 'base_model') and hasattr(model.base_model, 'gradient_checkpointing_enable'):
        model.base_model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled for memory efficiency")
    
    # Determine if using prompts
    use_prompt = args.use_prompt
    if not use_prompt and args.few_shot != 'none':
        print("Warning: --few_shot specified but --use_prompt not set. Ignoring few_shot.")
    
    # Create datasets with optional prompt formatting
    train_dataset = FineTuningDataset(
        train_texts, train_labels, tokenizer, args.max_length, 
        augment=True, use_prompt=use_prompt, source=args.source, few_shot=args.few_shot
    )
    val_dataset = FineTuningDataset(
        val_texts, val_labels, tokenizer, args.max_length,
        use_prompt=use_prompt, source=args.source, few_shot=args.few_shot
    )
    test_dataset = FineTuningDataset(
        test_texts, test_labels, tokenizer, args.max_length,
        use_prompt=use_prompt, source=args.source, few_shot=args.few_shot
    )
    
    if use_prompt:
        prompt_type = "few-shot" if args.few_shot != 'none' else "zero-shot"
        print(f"\nUsing {prompt_type} prompt-based fine-tuning")
        if args.few_shot != 'none':
            print(f"  Few-shot examples from: {args.few_shot}")
    else:
        print("\nUsing raw text fine-tuning (no prompts)")
    
    # Create dataloaders - use num_workers=0 for MPS to avoid multiprocessing issues
    num_workers = 0 if device.type == 'mps' else 2
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    
    if args.mode == 'train':
        # Train model
        print("\nStarting training...")
        model = train_model(
            model, train_loader, val_loader, device,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            source=args.source,
            model_name=args.model,
            use_focal_loss=args.use_focal_loss
        )
    
    # Evaluate model
    print("\nEvaluating model...")
    results = evaluate_model(
        model, test_loader, device, label_names,
        source=args.source,
        model_name=args.model
    )
    
    # Save results
    save_results(results, args.source, label_names, model_name=args.model, few_shot=args.few_shot)
    
    # Log completion if using nohup
    if not sys.stdout.isatty():
        print("-" * 80)
        print(f"Completed at: {datetime.datetime.now()}")

if __name__ == '__main__':
    main()
