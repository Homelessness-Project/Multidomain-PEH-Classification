#!/usr/bin/env python3
"""
SMOTE Analysis for Traditional ML Models

This script demonstrates the impact of SMOTE on SVM and Linear Regression
by comparing performance with and without SMOTE augmentation.

Usage:
    python scripts/smote_analysis.py --source reddit
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import os
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

def load_and_preprocess_data(source):
    """Load and preprocess data for a given source"""
    
    # Load soft labels
    soft_labels_path = f'output/annotation/soft_labels/{source}_soft_labels.csv'
    if not os.path.exists(soft_labels_path):
        raise FileNotFoundError(f"Soft labels file not found: {soft_labels_path}")
    
    soft_labels_df = pd.read_csv(soft_labels_path)
    
    # Load text data
    gold_standard_map = {
        'reddit': 'gold_standard/sampled_reddit_comments.csv',
        'x': 'gold_standard/sampled_twitter_posts.csv',
        'news': 'gold_standard/sampled_lexisnexis_news.csv',
        'meeting_minutes': 'gold_standard/sampled_meeting_minutes.csv'
    }
    
    if source not in gold_standard_map:
        raise ValueError(f"Unknown source: {source}")
    
    texts_df = pd.read_csv(gold_standard_map[source])
    
    # Merge data
    merged_df = texts_df.merge(soft_labels_df, on='id', how='inner')
    
    # Extract texts and labels
    texts = merged_df['text'].tolist()
    labels = merged_df[ALL_CATEGORIES].values.astype(float)
    
    print(f"Loaded {len(texts)} samples for {source}")
    print(f"Label distribution (mean): {np.mean(labels, axis=0)}")
    
    return texts, labels, ALL_CATEGORIES

def apply_smote_to_texts(texts, labels, target_ratio=0.3):
    """Apply SMOTE to text data using TF-IDF features"""
    
    print(f"\nApplying SMOTE to text data...")
    print(f"Original dataset size: {len(texts)}")
    
    # Convert texts to TF-IDF features
    vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 2))
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
                    
                    # Convert back to text (simplified - using original texts as templates)
                    new_texts = []
                    for j in range(len(new_texts), len(X_resampled)):
                        # For simplicity, we'll use the original text with some variation
                        # In practice, you might want to use more sophisticated text generation
                        original_idx = j % len(texts)
                        new_texts.append(texts[original_idx])
                    
                    # Add new samples
                    augmented_texts.extend(new_texts)
                    augmented_labels = np.vstack([augmented_labels, new_labels])
                    
                    print(f"    Added {len(new_texts)} samples for {category}")
                    
            except Exception as e:
                print(f"    Failed to augment {category}: {e}")
    
    print(f"Final dataset size: {len(augmented_texts)}")
    return augmented_texts, augmented_labels

def train_model_with_smote(X_train, y_train, X_test, y_test, model_name, model_class, use_smote=True):
    """Train a model with or without SMOTE"""
    
    if use_smote:
        print(f"\nTraining {model_name} WITH SMOTE...")
        # Apply SMOTE to the TF-IDF features
        smote = SMOTE(random_state=42, sampling_strategy=0.3)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        print(f"  Original training size: {X_train.shape[0]}")
        print(f"  SMOTE training size: {X_train_smote.shape[0]}")
    else:
        print(f"\nTraining {model_name} WITHOUT SMOTE...")
        X_train_smote, y_train_smote = X_train, y_train
    
    # Convert to binary for sklearn
    y_train_binary = (y_train_smote > 0.5).astype(int)
    y_test_binary = (y_test > 0.5).astype(int)
    
    # Train model
    model = MultiOutputClassifier(
        model_class(random_state=42, class_weight='balanced')
    )
    model.fit(X_train_smote, y_train_binary)
    
    # Evaluate
    y_pred = model.predict(X_test)
    macro_f1 = f1_score(y_test_binary, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_test_binary, y_pred, average='micro', zero_division=0)
    
    print(f"  {model_name} - Macro F1: {macro_f1:.4f}, Micro F1: {micro_f1:.4f}")
    
    return {
        'model_name': model_name,
        'use_smote': use_smote,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'predictions': y_pred
    }

def main():
    parser = argparse.ArgumentParser(description='SMOTE Analysis for Traditional ML')
    parser.add_argument('--source', type=str, required=True,
                       choices=['reddit', 'x', 'news', 'meeting_minutes'],
                       help='Data source to use')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    print(f"🔍 SMOTE ANALYSIS FOR {args.source.upper()}")
    print("="*60)
    
    # Load data
    texts, labels, label_names = load_and_preprocess_data(args.source)
    
    # Split data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=args.test_size, random_state=args.seed
    )
    
    print(f"Train: {len(train_texts)}, Test: {len(test_texts)}")
    
    # Convert to TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts).toarray()
    X_test = vectorizer.transform(test_texts).toarray()
    
    # Convert labels to binary for SMOTE
    y_train_binary = (train_labels > 0.5).astype(int)
    y_test_binary = (test_labels > 0.5).astype(int)
    
    # Test different models
    models_to_test = [
        ('Logistic Regression', LogisticRegression),
        ('SVM', SVC),
        ('Random Forest', RandomForestClassifier)
    ]
    
    results = []
    
    for model_name, model_class in models_to_test:
        # Test without SMOTE
        result_no_smote = train_model_with_smote(
            X_train, y_train_binary, X_test, y_test_binary, 
            model_name, model_class, use_smote=False
        )
        results.append(result_no_smote)
        
        # Test with SMOTE
        result_with_smote = train_model_with_smote(
            X_train, y_train_binary, X_test, y_test_binary, 
            model_name, model_class, use_smote=True
        )
        results.append(result_with_smote)
    
    # Create comparison table
    print(f"\n📊 SMOTE IMPACT COMPARISON")
    print("="*60)
    print(f"{'Model':<20} {'SMOTE':<8} {'Macro F1':<10} {'Micro F1':<10} {'Improvement':<12}")
    print("-" * 60)
    
    for i in range(0, len(results), 2):
        no_smote = results[i]
        with_smote = results[i+1]
        
        macro_improvement = with_smote['macro_f1'] - no_smote['macro_f1']
        micro_improvement = with_smote['micro_f1'] - no_smote['micro_f1']
        
        print(f"{no_smote['model_name']:<20} {'No':<8} {no_smote['macro_f1']:<10.4f} {no_smote['micro_f1']:<10.4f} {'':<12}")
        print(f"{'':<20} {'Yes':<8} {with_smote['macro_f1']:<10.4f} {with_smote['micro_f1']:<10.4f} {f'+{macro_improvement:.3f}':<12}")
        print()
    
    # Save results
    results_df = pd.DataFrame(results)
    output_dir = f'nlp_outputs/{args.source}'
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(f'{output_dir}/smote_analysis.csv', index=False)
    
    print(f"📁 Results saved to {output_dir}/smote_analysis.csv")

if __name__ == "__main__":
    main()

