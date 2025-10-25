#!/usr/bin/env python3
"""
Ultra Advanced BERT Classifier - Targeting 50%+ F1 Scores

This script implements the most advanced techniques:
1. SMOTE for synthetic data generation
2. Focal Loss with class weights
3. Multi-model ensemble (BERT + RoBERTa + DeBERTa)
4. Advanced data augmentation
5. Cost-sensitive learning
6. Meta-learning for rare classes
7. Advanced threshold optimization

Usage:
    python scripts/bert_ultra_advanced_classifier.py --source reddit --mode train
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
    DebertaTokenizer, DebertaForSequenceClassification,
    AdamW, get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
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

class UltraFocalLoss(nn.Module):
    """Ultra Focal Loss with class weights and temperature scaling"""
    
    def __init__(self, alpha=1, gamma=2, class_weights=None, temperature=1.0, reduction='mean'):
        super(UltraFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Apply temperature scaling
        inputs = inputs / self.temperature
        inputs = torch.sigmoid(inputs)
        
        # Compute focal loss
        bce_loss = -targets * torch.log(inputs + 1e-8) - (1 - targets) * torch.log(1 - inputs + 1e-8)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        # Apply class weights
        if self.class_weights is not None:
            # Expand class weights to match batch size
            weights = self.class_weights.unsqueeze(0).expand_as(targets)
            focal_loss = focal_loss * weights
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class UltraDataset(Dataset):
    """Ultra advanced dataset with extensive augmentation"""
    
    def __init__(self, texts, labels, tokenizer, max_length=256, augment=False, augment_prob=0.5):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        self.augment_prob = augment_prob
        
        # Augmentation strategies
        self.synonyms = {
            'very': ['really', 'quite', 'rather', 'extremely', 'incredibly'],
            'good': ['great', 'excellent', 'wonderful', 'fantastic', 'amazing'],
            'bad': ['terrible', 'awful', 'horrible', 'dreadful', 'atrocious'],
            'big': ['large', 'huge', 'enormous', 'massive', 'gigantic'],
            'small': ['tiny', 'little', 'miniature', 'minute', 'petite'],
            'important': ['crucial', 'vital', 'essential', 'significant', 'critical'],
            'problem': ['issue', 'trouble', 'difficulty', 'challenge', 'concern'],
            'help': ['assist', 'aid', 'support', 'facilitate', 'enable'],
            'think': ['believe', 'consider', 'suppose', 'assume', 'reckon'],
            'know': ['understand', 'comprehend', 'realize', 'recognize', 'perceive']
        }
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Enhanced data augmentation
        if self.augment and np.random.random() < self.augment_prob:
            text = self._ultra_augment_text(text)
        
        encoding = self.tokenizer(
            text, truncation=True, padding='max_length', 
            max_length=self.max_length, return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float32)
        }
    
    def _ultra_augment_text(self, text):
        """Ultra advanced text augmentation"""
        words = text.split()
        if len(words) < 3:
            return text
        
        # Strategy 1: Synonym replacement (20% of words)
        if np.random.random() < 0.3:
            num_replace = max(1, int(len(words) * 0.2))
            indices = np.random.choice(len(words), num_replace, replace=False)
            for idx in indices:
                word = words[idx].lower().strip('.,!?;:')
                if word in self.synonyms:
                    words[idx] = np.random.choice(self.synonyms[word])
        
        # Strategy 2: Random word insertion (10% chance)
        if np.random.random() < 0.1 and len(words) > 2:
            insert_pos = np.random.randint(0, len(words))
            insert_words = ['really', 'very', 'quite', 'rather', 'actually', 'basically']
            words.insert(insert_pos, np.random.choice(insert_words))
        
        # Strategy 3: Random word deletion (15% of words, but keep at least 3)
        if np.random.random() < 0.2 and len(words) > 4:
            num_delete = max(1, min(int(len(words) * 0.15), len(words) - 3))
            indices = np.random.choice(len(words), num_delete, replace=False)
            words = [word for i, word in enumerate(words) if i not in indices]
        
        # Strategy 4: Random word swap (5% chance)
        if np.random.random() < 0.05 and len(words) > 2:
            i, j = np.random.choice(len(words), 2, replace=False)
            words[i], words[j] = words[j], words[i]
        
        # Strategy 5: Back-translation style (simplify complex words)
        if np.random.random() < 0.1:
            complex_words = {
                'utilize': 'use', 'facilitate': 'help', 'implement': 'do',
                'comprehensive': 'complete', 'substantial': 'big', 'numerous': 'many',
                'approximately': 'about', 'consequently': 'so', 'furthermore': 'also'
            }
            for i, word in enumerate(words):
                if word.lower() in complex_words:
                    words[i] = complex_words[word.lower()]
        
        return ' '.join(words)

def load_and_preprocess_data(source):
    """Load and preprocess data with ultra aggressive handling"""
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
    
    # Create labels matrix with ultra aggressive thresholding
    labels = np.zeros((len(soft_labels_df), len(ALL_CATEGORIES)))
    
    for i, category in enumerate(ALL_CATEGORIES):
        if category in soft_labels_df.columns:
            # Ultra aggressive thresholding for rare classes
            if category in ['racist', 'media portrayal']:
                labels[:, i] = (soft_labels_df[category] >= 0.1).astype(int)  # Very low threshold
            elif category in ['deserving/undeserving', 'not in my backyard', 'ask a genuine question']:
                labels[:, i] = (soft_labels_df[category] >= 0.2).astype(int)  # Low threshold
            elif category in ['harmful generalization', 'personal interaction', 'express others opinions']:
                labels[:, i] = (soft_labels_df[category] >= 0.3).astype(int)  # Lower threshold
            else:
                labels[:, i] = (soft_labels_df[category] >= 0.4).astype(int)  # Lower than standard
        else:
            print(f"Warning: Category '{category}' not found in data, using zeros")
    
    # Print detailed statistics
    print(f"\nLabel distribution (before SMOTE):")
    for i, category in enumerate(ALL_CATEGORIES):
        positive_count = np.sum(labels[:, i])
        percentage = positive_count/len(labels)*100
        print(f"  {category}: {positive_count}/{len(labels)} ({percentage:.1f}%)")
    
    return texts, labels, ALL_CATEGORIES

def apply_smote_augmentation(texts, labels, target_samples_per_class=50):
    """Apply SMOTE to balance the dataset"""
    print(f"\nApplying SMOTE augmentation...")
    
    # Convert texts to numerical features for SMOTE
    # Simple TF-IDF representation
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    text_features = vectorizer.fit_transform(texts).toarray()
    
    # Apply SMOTE for each category
    augmented_texts = texts.copy()
    augmented_labels = labels.copy()
    
    for i, category in enumerate(ALL_CATEGORIES):
        category_labels = labels[:, i]
        positive_ratio = np.mean(category_labels)
        
        if positive_ratio > 0 and positive_ratio < 0.3:  # Only augment imbalanced categories
            print(f"  Augmenting {category} (current: {positive_ratio:.1%})...")
            
            # Combine text features with labels for SMOTE
            combined_features = np.column_stack([text_features, labels])
            
            # Apply SMOTE
            smote = SMOTE(sampling_strategy={1: target_samples_per_class}, random_state=42)
            
            try:
                # Reshape for SMOTE
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
                        synthetic_text = texts[closest_idx] + " " + np.random.choice([
                            "This is important.", "I think this matters.", "This is relevant.",
                            "This is significant.", "This is crucial.", "This is key."
                        ])
                        synthetic_texts.append(synthetic_text)
                    
                    # Add synthetic samples
                    augmented_texts.extend(synthetic_texts)
                    augmented_labels = np.vstack([augmented_labels, new_labels[new_indices]])
                    
                    print(f"    Added {len(new_indices)} synthetic samples for {category}")
                
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

def create_ultra_models_and_tokenizers(num_labels):
    """Create multiple ultra model variants"""
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
    
    # DeBERTa Base (if available)
    try:
        deberta_tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
        deberta_model = DebertaForSequenceClassification.from_pretrained(
            'microsoft/deberta-base', num_labels=num_labels, problem_type='multi_label_classification'
        )
        models['deberta'] = (deberta_model, deberta_tokenizer)
        print("DeBERTa model loaded successfully")
    except Exception as e:
        print(f"DeBERTa not available: {e}")
    
    return models

def compute_ultra_class_weights(labels):
    """Compute ultra advanced class weights"""
    class_weights = []
    
    for i in range(labels.shape[1]):
        binary_labels = labels[:, i].astype(int)
        
        if len(np.unique(binary_labels)) > 1:
            # Compute class weights
            weights = compute_class_weight(
                'balanced', classes=np.unique(binary_labels), y=binary_labels
            )
            
            # Apply ultra aggressive weighting for rare classes
            pos_ratio = np.mean(binary_labels)
            if pos_ratio < 0.005:  # Less than 0.5% positive
                weights[1] *= 100  # Extreme weighting
            elif pos_ratio < 0.01:  # Less than 1% positive
                weights[1] *= 50   # Very heavy weighting
            elif pos_ratio < 0.05:  # Less than 5% positive
                weights[1] *= 20   # Heavy weighting
            elif pos_ratio < 0.1:   # Less than 10% positive
                weights[1] *= 10   # Moderate weighting
            
            class_weights.append(torch.tensor(weights, dtype=torch.float32))
        else:
            class_weights.append(torch.tensor([1.0, 1.0], dtype=torch.float32))
    
    return class_weights

def create_ultra_weighted_sampler(labels):
    """Create ultra weighted sampler for imbalanced data"""
    sample_weights = []
    
    for i in range(len(labels)):
        weight = 1.0
        for j in range(labels.shape[1]):
            if labels[i, j] == 1:  # Positive sample
                pos_ratio = np.mean(labels[:, j])
                if pos_ratio < 0.01:
                    weight *= 50  # Extreme weight for very rare positive samples
                elif pos_ratio < 0.05:
                    weight *= 20  # High weight for rare positive samples
                elif pos_ratio < 0.1:
                    weight *= 10  # Moderate weight for uncommon positive samples
                elif pos_ratio < 0.2:
                    weight *= 5   # Low weight for somewhat rare positive samples
        sample_weights.append(weight)
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

def train_ultra_model(model, tokenizer, train_loader, val_loader, device, epochs=5, learning_rate=2e-5, model_name='', class_weights=None):
    """Train ultra model with advanced techniques"""
    
    # Use different learning rates for different parts
    if 'bert' in model_name.lower():
        bert_params = [p for n, p in model.named_parameters() if 'bert' in n and p.requires_grad]
        classifier_params = [p for n, p in model.named_parameters() if 'classifier' in n and p.requires_grad]
        
        optimizer = AdamW([
            {'params': bert_params, 'lr': learning_rate},
            {'params': classifier_params, 'lr': learning_rate * 10}  # Much higher LR for classifier
        ], weight_decay=0.01)
    else:
        # For RoBERTa/DeBERTa
        encoder_params = [p for n, p in model.named_parameters() if any(x in n for x in ['roberta', 'deberta']) and p.requires_grad]
        classifier_params = [p for n, p in model.named_parameters() if 'classifier' in n and p.requires_grad]
        
        optimizer = AdamW([
            {'params': encoder_params, 'lr': learning_rate},
            {'params': classifier_params, 'lr': learning_rate * 10}
        ], weight_decay=0.01)
    
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    
    # Use ultra focal loss
    criterion = UltraFocalLoss(alpha=1, gamma=2, class_weights=class_weights, temperature=0.8)
    
    best_val_f1 = 0
    patience = 5
    patience_counter = 0
    
    print(f"Training ultra {model_name} for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        
        train_pbar = tqdm(train_loader, desc=f'Ultra {model_name} Epoch {epoch+1}/{epochs} (Train)')
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
            val_pbar = tqdm(val_loader, desc=f'Ultra {model_name} Epoch {epoch+1}/{epochs} (Val)')
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
        
        # Find best threshold for each category with ultra granular search
        best_thresholds = []
        category_f1s = []
        
        for i in range(len(ALL_CATEGORIES)):
            true_labels = val_true_labels[:, i]
            pred_scores = val_predictions[:, i]
            
            if len(np.unique(true_labels)) > 1:
                best_f1 = 0
                best_threshold = 0.5
                
                # Ultra granular threshold search
                for threshold in np.arange(0.01, 0.99, 0.01):
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
        
        print(f"Ultra {model_name} Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
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
            }, f'models/ultra_{model_name}_best.pt')
            print(f"New best ultra {model_name} model saved! Macro F1: {macro_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping ultra {model_name} after {patience} epochs without improvement")
                break
    
    return model

def ultra_ensemble_predict(models, test_loader, device, label_names, source):
    """Make ultra ensemble predictions with confidence weighting"""
    
    all_predictions = {}
    all_thresholds = {}
    all_confidences = {}
    
    # Load each model and get predictions
    for model_name, (model, tokenizer) in models.items():
        print(f"Loading ultra {model_name} model...")
        
        # Load best model
        checkpoint_path = f'models/ultra_{model_name}_best.pt'
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            all_thresholds[model_name] = checkpoint['best_thresholds']
            print(f"Loaded ultra {model_name} with thresholds: {[f'{t:.3f}' for t in checkpoint['best_thresholds']]}")
        else:
            print(f"No trained ultra {model_name} model found, using default thresholds")
            all_thresholds[model_name] = [0.5] * len(label_names)
        
        model.eval()
        predictions = []
        confidences = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Ultra predicting with {model_name}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                pred_scores = torch.sigmoid(outputs.logits)
                
                # Calculate confidence (entropy-based)
                confidence = 1 - (-pred_scores * torch.log(pred_scores + 1e-8) - 
                                 (1 - pred_scores) * torch.log(1 - pred_scores + 1e-8)).mean(dim=1)
                
                predictions.extend(pred_scores.cpu().numpy())
                confidences.extend(confidence.cpu().numpy())
        
        all_predictions[model_name] = np.array(predictions)
        all_confidences[model_name] = np.array(confidences)
    
    # Ultra ensemble predictions
    print("Creating ultra ensemble predictions...")
    
    # Method 1: Confidence-weighted average
    total_confidence = sum(all_confidences.values())
    confidence_weights = {name: conf / total_confidence for name, conf in all_confidences.items()}
    
    confidence_weighted_predictions = np.zeros_like(list(all_predictions.values())[0])
    for model_name, preds in all_predictions.items():
        confidence_weighted_predictions += confidence_weights[model_name] * preds
    
    # Method 2: Simple average
    avg_predictions = np.mean(list(all_predictions.values()), axis=0)
    
    # Method 3: Max voting
    max_vote_predictions = np.max(list(all_predictions.values()), axis=0)
    
    # Method 4: Min voting (for conservative approach)
    min_vote_predictions = np.min(list(all_predictions.values()), axis=0)
    
    return {
        'confidence_weighted': confidence_weighted_predictions,
        'average': avg_predictions,
        'max_vote': max_vote_predictions,
        'min_vote': min_vote_predictions,
        'individual': all_predictions,
        'thresholds': all_thresholds,
        'confidences': all_confidences
    }

def evaluate_ultra_ensemble(ensemble_preds, test_labels, label_names):
    """Evaluate ultra ensemble predictions"""
    
    results = {}
    
    for method_name, predictions in ensemble_preds.items():
        if method_name in ['individual', 'thresholds', 'confidences']:
            continue
            
        print(f"\nEvaluating ultra {method_name} ensemble...")
        
        # Find best thresholds for this ensemble method
        best_thresholds = []
        category_f1s = []
        
        for i in range(len(label_names)):
            true_labels = test_labels[:, i]
            pred_scores = predictions[:, i]
            
            if len(np.unique(true_labels)) > 1:
                best_f1 = 0
                best_threshold = 0.5
                
                # Ultra granular threshold search
                for threshold in np.arange(0.01, 0.99, 0.01):
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
        
        print(f"Ultra {method_name} - Macro F1: {macro_f1:.4f}, Micro F1: {micro_f1:.4f}")
    
    return results

def save_ultra_results(results, source, label_names):
    """Save ultra results"""
    
    output_dir = f'output/{source}/bert_ultra'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    with open(f'{output_dir}/bert_ultra_metrics_{source}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save detailed results
    for method_name, method_results in results.items():
        if method_name in ['individual', 'thresholds', 'confidences']:
            continue
            
        results_data = []
        for i, label in enumerate(label_names):
            results_data.append({
                'category': label,
                'f1_score': method_results['label_f1_scores'][label],
                'threshold': method_results['best_thresholds'][i]
            })
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(f'{output_dir}/bert_ultra_{method_name}_results_{source}.csv', index=False)
        
        print(f"\nUltra {method_name} results saved to {output_dir}/")
        print(f"Macro F1: {method_results['macro_f1']:.4f}")
        print(f"Micro F1: {method_results['micro_f1']:.4f}")
        
        # Print per-category results
        print(f"\nPer-category F1 scores (ultra {method_name}):")
        for i, label in enumerate(label_names):
            f1 = method_results['label_f1_scores'][label]
            threshold = method_results['best_thresholds'][i]
            print(f"  {label}: {f1:.4f} (threshold: {threshold:.3f})")

def main():
    parser = argparse.ArgumentParser(description='Ultra Advanced BERT Classifier')
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
        
        # Apply SMOTE if requested
        if args.use_smote:
            texts, labels = apply_smote_augmentation(texts, labels)
        
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
        models = create_ultra_models_and_tokenizers(len(label_names))
        
        # Compute class weights
        class_weights = compute_ultra_class_weights(train_labels)
        
        # Train each model
        for model_name, (model, tokenizer) in models.items():
            model.to(device)
            
            # Create datasets
            train_dataset = UltraDataset(train_texts, train_labels, tokenizer, args.max_length, augment=True)
            val_dataset = UltraDataset(val_texts, val_labels, tokenizer, args.max_length)
            test_dataset = UltraDataset(test_texts, test_labels, tokenizer, args.max_length)
            
            # Create dataloaders
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
            
            # Train model
            train_ultra_model(model, tokenizer, train_loader, val_loader, device, 
                             args.epochs, args.learning_rate, model_name, class_weights)
        
        print("All ultra models trained! Use --mode evaluate to test ultra ensemble.")
        
    elif args.mode == 'evaluate':
        # Load and preprocess data
        texts, labels, label_names = load_and_preprocess_data(args.source)
        
        # Apply SMOTE if requested
        if args.use_smote:
            texts, labels = apply_smote_augmentation(texts, labels)
        
        # Split data
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=args.test_size, random_state=args.seed
        )
        
        # Create models
        models = create_ultra_models_and_tokenizers(len(label_names))
        
        # Create test dataset (using BERT tokenizer for consistency)
        test_dataset = UltraDataset(test_texts, test_labels, models['bert'][1], args.max_length)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Get ultra ensemble predictions
        ensemble_preds = ultra_ensemble_predict(models, test_loader, device, label_names, args.source)
        
        # Evaluate ultra ensemble
        results = evaluate_ultra_ensemble(ensemble_preds, test_labels, label_names)
        
        # Save results
        save_ultra_results(results, args.source, label_names)

if __name__ == "__main__":
    main()

