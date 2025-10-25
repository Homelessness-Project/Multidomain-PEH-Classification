#!/usr/bin/env python3
"""
Debug BERT Predictions Script

This script examines the actual predictions to understand why F1 scores are so low.
"""

import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix
import json
import os

# Define all 16 categories
ALL_CATEGORIES = [
    'ask a genuine question', 'ask a rhetorical question', 'provide a fact or claim',
    'provide an observation', 'express their opinion', 'express others opinions',
    'money aid allocation', 'government critique', 'societal critique',
    'solutions/interventions', 'personal interaction', 'media portrayal',
    'not in my backyard', 'harmful generalization', 'deserving/undeserving', 'racist'
]

def load_model_and_data(source):
    """Load the trained model and test data"""
    
    # Load soft labels
    soft_labels_path = f'output/annotation/soft_labels/{source}_soft_labels.csv'
    soft_labels_df = pd.read_csv(soft_labels_path)
    
    # Load text data
    gold_standard_map = {
        'reddit': 'gold_standard/sampled_reddit_comments.csv',
        'x': 'gold_standard/sampled_twitter_posts.csv',
        'news': 'gold_standard/sampled_lexisnexis_news.csv',
        'meeting_minutes': 'gold_standard/sampled_meeting_minutes.csv'
    }
    
    text_file_path = gold_standard_map.get(source)
    text_df = pd.read_csv(text_file_path)
    
    # Get text column
    text_col_map = {
        'reddit': 'Comment', 'x': 'Deidentified_text', 
        'news': 'Deidentified_paragraph_text', 'meeting_minutes': 'Deidentified_paragraph'
    }
    
    text_col = text_col_map.get(source)
    texts = text_df[text_col].fillna('').astype(str).tolist()
    
    # Align data lengths
    if len(soft_labels_df) != len(text_df):
        min_len = min(len(soft_labels_df), len(text_df))
        soft_labels_df = soft_labels_df.iloc[:min_len]
        texts = texts[:min_len]
    
    # Create labels matrix
    labels = np.zeros((len(soft_labels_df), len(ALL_CATEGORIES)))
    for i, category in enumerate(ALL_CATEGORIES):
        if category in soft_labels_df.columns:
            labels[:, i] = (soft_labels_df[category] >= 0.5).astype(int)
    
    return texts, labels, soft_labels_df

def load_trained_model(source, model_path):
    """Load a trained model"""
    
    # Load model
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=len(ALL_CATEGORIES),
        problem_type='multi_label_classification'
    )
    
    # Load weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model not found: {model_path}")
        return None, None
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer

def analyze_predictions(model, tokenizer, texts, labels, source, model_name):
    """Analyze predictions in detail"""
    
    print(f"\n{'='*80}")
    print(f"ANALYZING {model_name.upper()} PREDICTIONS")
    print(f"{'='*80}")
    
    model.eval()
    all_predictions = []
    
    # Process in batches
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        encodings = tokenizer(
            batch_texts, 
            truncation=True, 
            padding=True, 
            max_length=256, 
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = model(
                input_ids=encodings['input_ids'],
                attention_mask=encodings['attention_mask']
            )
            predictions = torch.sigmoid(outputs.logits)
            all_predictions.extend(predictions.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    
    print(f"Prediction shape: {all_predictions.shape}")
    print(f"Label shape: {labels.shape}")
    
    # Analyze each category
    print(f"\n{'='*80}")
    print("CATEGORY ANALYSIS")
    print(f"{'='*80}")
    
    results = []
    
    for i, category in enumerate(ALL_CATEGORIES):
        true_labels = labels[:, i]
        pred_scores = all_predictions[:, i]
        
        # Count positive/negative samples
        pos_count = np.sum(true_labels)
        neg_count = len(true_labels) - pos_count
        
        print(f"\n{category}:")
        print(f"  Positive samples: {pos_count}/{len(true_labels)} ({pos_count/len(true_labels)*100:.1f}%)")
        print(f"  Negative samples: {neg_count}/{len(true_labels)} ({neg_count/len(true_labels)*100:.1f}%)")
        
        # Analyze prediction scores
        print(f"  Prediction score range: {pred_scores.min():.4f} - {pred_scores.max():.4f}")
        print(f"  Prediction score mean: {pred_scores.mean():.4f}")
        print(f"  Prediction score std: {pred_scores.std():.4f}")
        
        # Test different thresholds
        best_f1 = 0
        best_threshold = 0.5
        best_precision = 0
        best_recall = 0
        
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            binary_pred = (pred_scores > threshold).astype(int)
            
            if len(np.unique(true_labels)) > 1 and len(np.unique(binary_pred)) > 1:
                f1 = f1_score(true_labels, binary_pred, zero_division=0)
                precision, recall, _, _ = precision_recall_fscore_support(
                    true_labels, binary_pred, average='binary', zero_division=0
                )
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                    best_precision = precision
                    best_recall = recall
        
        print(f"  Best F1: {best_f1:.4f} at threshold {best_threshold}")
        print(f"  Best Precision: {best_precision:.4f}")
        print(f"  Best Recall: {best_recall:.4f}")
        
        # Confusion matrix at best threshold
        binary_pred = (pred_scores > best_threshold).astype(int)
        cm = confusion_matrix(true_labels, binary_pred)
        print(f"  Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
        
        results.append({
            'category': category,
            'pos_count': pos_count,
            'neg_count': neg_count,
            'pos_percentage': pos_count/len(true_labels)*100,
            'best_f1': best_f1,
            'best_threshold': best_threshold,
            'best_precision': best_precision,
            'best_recall': best_recall,
            'pred_mean': pred_scores.mean(),
            'pred_std': pred_scores.std()
        })
    
    return results

def main():
    source = 'reddit'
    
    # Load data
    texts, labels, soft_labels_df = load_model_and_data(source)
    print(f"Loaded {len(texts)} samples")
    
    # Test different models
    models_to_test = [
        ('original', f'models/bert_best_{source}.pt'),
        ('frozen', f'models/bert_frozen_best_{source}.pt'),
        ('improved', f'models/bert_improved_best_{source}.pt')
    ]
    
    all_results = {}
    
    for model_name, model_path in models_to_test:
        model, tokenizer = load_trained_model(source, model_path)
        if model is not None:
            results = analyze_predictions(model, tokenizer, texts, labels, source, model_name)
            all_results[model_name] = results
    
    # Create summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")
    
    summary_data = []
    for model_name, results in all_results.items():
        for result in results:
            summary_data.append({
                'Model': model_name,
                'Category': result['category'],
                'Pos %': f"{result['pos_percentage']:.1f}%",
                'F1': f"{result['best_f1']:.4f}",
                'Threshold': f"{result['best_threshold']:.2f}",
                'Precision': f"{result['best_precision']:.4f}",
                'Recall': f"{result['best_recall']:.4f}"
            })
    
    df = pd.DataFrame(summary_data)
    print(df.to_string(index=False))
    
    # Save results
    os.makedirs('output/debug_analysis', exist_ok=True)
    df.to_csv('output/debug_analysis/prediction_analysis.csv', index=False)
    
    print(f"\nResults saved to output/debug_analysis/prediction_analysis.csv")

if __name__ == "__main__":
    main()

