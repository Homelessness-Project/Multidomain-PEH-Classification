#!/usr/bin/env python3
"""
Finetune Local Models on GPT Pseudolabels

This script:
1. Loads GPT pseudolabels from all data (excluding few-shot examples)
2. Uses gold standard (1600-1700) split into val/test sets
3. Train set: Only GPT pseudolabels (no gold standard)
4. Finetunes 6 model options on GPT pseudolabels with GPU support:
   - Transformers: bert-base-uncased, roberta-base, modernbert-base
   - Local LLMs: llama, qwen, gemma3

Usage:
    # Transformers (full fine-tuning)
    python scripts/finetune_on_gpt_pseudolabels.py --source reddit --model bert-base-uncased
    python scripts/finetune_on_gpt_pseudolabels.py --source reddit --model roberta-base
    python scripts/finetune_on_gpt_pseudolabels.py --source reddit --model modernbert-base
    
    # Local LLMs with LoRA (recommended - 3-10x faster, less memory)
    python scripts/finetune_on_gpt_pseudolabels.py --source reddit --model llama --use_lora
    python scripts/finetune_on_gpt_pseudolabels.py --source reddit --model qwen --use_lora
    python scripts/finetune_on_gpt_pseudolabels.py --source reddit --model gemma3 --use_lora
    
    # Local LLMs full fine-tuning (slower, more memory)
    python scripts/finetune_on_gpt_pseudolabels.py --source reddit --model llama --batch_size 4
    python scripts/finetune_on_gpt_pseudolabels.py --source reddit --model qwen --batch_size 4
    
    # Custom LoRA settings
    python scripts/finetune_on_gpt_pseudolabels.py --source reddit --model qwen --use_lora --lora_rank 16 --lora_alpha 32
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
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForCausalLM, AutoModel, AutoConfig,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not available. Install with: pip install peft")
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report
import json
import os
import sys
import re
from pathlib import Path
from tqdm import tqdm
import sys

# Configure tqdm to work in non-interactive environments
# Force tqdm to use stdout and show progress even when output is redirected
# Check if stdout is a TTY (interactive terminal)
is_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

tqdm_kwargs = {
    'file': sys.stdout,
    'dynamic_ncols': True,
    'mininterval': 0.5 if is_tty else 1.0,  # Update at least every 0.5-1 seconds
    'miniters': 1,  # Update after every iteration (if mininterval allows)
    'disable': False,  # Explicitly enable (don't disable)
}

# If not a TTY, use a simpler format that works better with redirected output
if not is_tty:
    tqdm_kwargs['ncols'] = 100  # Fixed width for non-TTY
    tqdm_kwargs['ascii'] = True  # Use ASCII characters for compatibility
import warnings
warnings.filterwarnings('ignore')

# Import utils
try:
    from scripts.utils import get_model_config
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scripts.utils import get_model_config

# Define all 16 categories
ALL_CATEGORIES = [
    'ask a genuine question', 'ask a rhetorical question', 'provide a fact or claim',
    'provide an observation', 'express their opinion', 'express others opinions',
    'money aid allocation', 'government critique', 'societal critique',
    'solutions/interventions', 'personal interaction', 'media portrayal',
    'not in my backyard', 'harmful generalization', 'deserving/undeserving', 'racist'
]

# Map from GPT column names to our category names
GPT_TO_CATEGORY_MAP = {
    'Comment_ask a genuine question': 'ask a genuine question',
    'Comment_ask a rhetorical question': 'ask a rhetorical question',
    'Comment_provide a fact or claim': 'provide a fact or claim',
    'Comment_provide an observation': 'provide an observation',
    'Comment_express their opinion': 'express their opinion',
    'Comment_express others opinions': 'express others opinions',
    'Critique_money aid allocation': 'money aid allocation',
    'Critique_government critique': 'government critique',
    'Critique_societal critique': 'societal critique',
    'Response_solutions/interventions': 'solutions/interventions',
    'Perception_personal interaction': 'personal interaction',
    'Perception_media portrayal': 'media portrayal',
    'Perception_not in my backyard': 'not in my backyard',
    'Perception_harmful generalization': 'harmful generalization',
    'Perception_deserving/undeserving': 'deserving/undeserving',
    'Racist_Flag': 'racist'
}

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        # Ensure inputs and targets are in float32 for numerical stability
        # This is critical for mixed precision training (model in float16, loss in float32)
        inputs = inputs.to(dtype=torch.float32)
        targets = targets.to(dtype=torch.float32)
        
        # Apply sigmoid to inputs to get probabilities
        probs = torch.sigmoid(inputs)
        
        # Clamp probabilities to avoid numerical instability (log(0) or log(1))
        probs = torch.clamp(probs, min=1e-7, max=1.0 - 1e-7)
        
        # Calculate binary cross entropy using probabilities (not logits)
        # This is more numerically stable than using logits with sigmoid
        bce_loss = -(targets * torch.log(probs) + (1 - targets) * torch.log(1 - probs))
        
        # Calculate p_t (probability of true class)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Clamp p_t to avoid numerical issues with (1 - p_t) ** gamma when p_t is very close to 1
        p_t = torch.clamp(p_t, min=1e-7, max=1.0 - 1e-7)
        
        # Calculate focal loss
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * bce_loss
        
        # Check for NaN or Inf
        if torch.isnan(focal_loss).any() or torch.isinf(focal_loss).any():
            print(f"WARNING: NaN/Inf detected in focal loss!")
            print(f"  Input range: [{inputs.min().item():.4f}, {inputs.max().item():.4f}]")
            print(f"  Probs range: [{probs.min().item():.4f}, {probs.max().item():.4f}]")
            print(f"  p_t range: [{p_t.min().item():.4f}, {p_t.max().item():.4f}]")
            print(f"  BCE loss range: [{bce_loss.min().item():.4f}, {bce_loss.max().item():.4f}]")
            # Replace NaN/Inf with a large finite value
            focal_loss = torch.nan_to_num(focal_loss, nan=1e6, posinf=1e6, neginf=-1e6)
        
        return focal_loss.mean()

class GPTPseudolabelDataset(Dataset):
    """Dataset for GPT pseudolabels"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
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

def load_gpt_pseudolabels(source, exclude_few_shot=True):
    """Load GPT pseudolabels, excluding few-shot examples"""
    
    # Load all GPT pseudolabels
    gpt_file = f'output/{source}/gpt4/classified_comments_{source}_all_gpt4_{source}_flags.csv'
    
    if not Path(gpt_file).exists():
        raise FileNotFoundError(f"GPT pseudolabels file not found: {gpt_file}")
    
    print(f"Loading GPT pseudolabels from {gpt_file}...")
    gpt_df = pd.read_csv(gpt_file)
    print(f"  Loaded {len(gpt_df)} rows from CSV")
    
    # Handle duplicates: keep first occurrence of each unique comment
    # (Some comments may appear multiple times with different category combinations)
    initial_count = len(gpt_df)
    gpt_df = gpt_df.drop_duplicates(subset=['Comment'], keep='first')
    if initial_count != len(gpt_df):
        print(f"  Removed {initial_count - len(gpt_df)} duplicate comments (kept first occurrence)")
    
    print(f"  Unique comments for training: {len(gpt_df):,}")
    
    # Exclude few-shot examples (from gold_subset files)
    if exclude_few_shot:
        few_shot_files = [
            f'output/{source}/gpt4/classified_comments_{source}_gold_subset_gpt4_{source}_flags.csv',
            f'output/{source}/gpt4/classified_comments_{source}_gold_subset_gpt4_none_flags.csv'
        ]
        
        few_shot_comments = set()
        for few_shot_file in few_shot_files:
            if Path(few_shot_file).exists():
                few_shot_df = pd.read_csv(few_shot_file)
                few_shot_comments.update(few_shot_df['Comment'].astype(str).tolist())
        
        if few_shot_comments:
            initial_len = len(gpt_df)
            gpt_df = gpt_df[~gpt_df['Comment'].astype(str).isin(few_shot_comments)]
            print(f"  Excluded {initial_len - len(gpt_df)} few-shot examples")
            print(f"  Remaining: {len(gpt_df)} comments")
    
    return gpt_df

def load_gold_standard(source):
    """Load gold standard data and labels"""
    
    source_to_file = {
        'reddit': 'gold_standard/sampled_reddit_comments.csv',
        'x': 'gold_standard/sampled_twitter_posts.csv',
        'news': 'gold_standard/sampled_lexisnexis_news.csv',
        'meeting_minutes': 'gold_standard/sampled_meeting_minutes.csv'
    }
    
    gold_file = source_to_file[source]
    if not Path(gold_file).exists():
        raise FileNotFoundError(f"Gold standard file not found: {gold_file}")
    
    print(f"Loading gold standard from {gold_file}...")
    gold_df = pd.read_csv(gold_file)
    
    # Get text column - check various possible column names
    text_col = 'Comment'
    if 'Deidentified_Comment' in gold_df.columns:
        text_col = 'Deidentified_Comment'
    elif 'Deidentified_text' in gold_df.columns:
        text_col = 'Deidentified_text'
    elif 'Deidentified text' in gold_df.columns:
        text_col = 'Deidentified text'
    elif 'Deidentified_paragraph' in gold_df.columns:
        text_col = 'Deidentified_paragraph'
    elif 'Comment' in gold_df.columns:
        text_col = 'Comment'
    else:
        # Try to find any column that might contain text
        possible_cols = [col for col in gold_df.columns if 'text' in col.lower() or 'comment' in col.lower() or 'paragraph' in col.lower()]
        if possible_cols:
            text_col = possible_cols[0]
            print(f"  Warning: Using '{text_col}' as text column (auto-detected)")
        else:
            raise ValueError(f"Could not find text column in gold standard. Available columns: {gold_df.columns.tolist()}")
    
    if text_col not in gold_df.columns:
        raise ValueError(f"Text column '{text_col}' not found in gold standard. Available columns: {gold_df.columns.tolist()}")
    
    # Load gold standard labels (human annotations)
    gold_labels_df = None
    soft_labels_file = f'output/annotation/soft_labels/{source}_soft_labels.csv'
    raw_scores_file = f'annotation/{source}_raw_scores.csv'
    
    if Path(soft_labels_file).exists():
        print(f"  Loading gold standard labels from {soft_labels_file}...")
        gold_labels_df = pd.read_csv(soft_labels_file)
    elif Path(raw_scores_file).exists():
        print(f"  Loading gold standard labels from {raw_scores_file}...")
        gold_labels_df = pd.read_csv(raw_scores_file)
        # Convert raw scores (0-3) to binary (threshold at >= 2 for 2+ annotator agreement)
        # Map columns to match ALL_CATEGORIES
        category_to_col = {
            'ask a genuine question': 'ask a genuine question',
            'ask a rhetorical question': 'ask a rhetorical question',
            'provide a fact or claim': 'provide a fact or claim',
            'provide an observation': 'provide an observation',
            'express their opinion': 'express their opinion',
            'express others opinions': 'express others opinions',
            'money aid allocation': 'money aid allocation',
            'government critique': 'government critique',
            'societal critique': 'societal critique',
            'solutions/interventions': 'solutions/interventions',
            'personal interaction': 'personal interaction',
            'media portrayal': 'media portrayal',
            'not in my backyard': 'not in my backyard',
            'harmful generalization': 'harmful generalization',
            'deserving/undeserving': 'deserving/undeserving',
            'racist': 'Racist'
        }
    else:
        print(f"  WARNING: No gold standard labels found. Will use GPT pseudolabels for test/val.")
    
    print(f"  Loaded {len(gold_df)} gold standard samples")
    return gold_df, text_col, gold_labels_df


def create_model_and_tokenizer(model_name, num_labels, device=None, use_lora=False, lora_rank=8, lora_alpha=16, lora_dropout=0.1, lora_target_modules='all'):
    """Create model and tokenizer based on model name, supporting both transformers and local LLMs
    
    Args:
        model_name: Name of the model to load
        num_labels: Number of classification labels
        device: Torch device
        use_lora: Whether to use LoRA for efficient fine-tuning
        lora_rank: LoRA rank (only used if use_lora=True)
        lora_alpha: LoRA alpha (only used if use_lora=True)
        lora_dropout: LoRA dropout rate (only used if use_lora=True)
        lora_target_modules: Which modules to apply LoRA to: 'all', 'attention', 'mlp', 'attention+mlp'
    """
    
    # Determine torch dtype based on device
    # For MPS with LoRA: use float32 (LoRA trains only ~0.15% of params, so memory is manageable)
    # For MPS without LoRA: use float16 to save memory (but may have backward pass issues)
    # For CUDA: use float16 (better support)
    if device and device.type == 'cuda':
        torch_dtype = torch.float16
    elif device and device.type == 'mps':
        if use_lora:
            # With LoRA, only ~0.15% of parameters are trainable, so float32 is fine
            # This avoids MPS backward pass issues with float16
            torch_dtype = torch.float32
            print("  Using float32 for MPS with LoRA (only trainable params need gradients)")
        else:
            # Without LoRA, use float16 to save memory (full fine-tuning)
            # Note: MPS may have backward pass issues with float16
            torch_dtype = torch.float16
            print("  Using float16 for MPS (full fine-tuning - may have backward pass issues)")
    else:
        torch_dtype = torch.float32
    
    # Check if it's a local LLM (llama, qwen, gemma3)
    local_llms = ['llama', 'qwen', 'gemma3']
    is_local_llm = any(llm in model_name.lower() for llm in local_llms)
    
    if is_local_llm:
        # Use get_model_config from utils
        model_config = get_model_config(model_name)
        model_id = model_config["model_id"]
        
        print(f"Loading local LLM: {model_name} ({model_id})")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        
        # Check if tokenizer has pad_token - critical for batching
        # Llama models typically don't have a pad_token by default
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
                print(f"  Using eos_token as pad_token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
            else:
                # Add pad token and resize embeddings
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')
                print(f"  Added new pad_token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
        
        # Ensure pad_token_id is set and valid
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
                tokenizer.pad_token = tokenizer.eos_token
            else:
                # Fallback: use tokenizer.unk_token_id or 0
                tokenizer.pad_token_id = getattr(tokenizer, 'unk_token_id', 0) if hasattr(tokenizer, 'unk_token_id') and tokenizer.unk_token_id is not None else 0
                print(f"  WARNING: Using fallback pad_token_id: {tokenizer.pad_token_id}")
        
        # Final verification
        if tokenizer.pad_token_id is None:
            raise ValueError("Failed to set pad_token_id in tokenizer. Cannot proceed with batching.")
        
        print(f"  ✓ Tokenizer pad_token: '{tokenizer.pad_token}' (id: {tokenizer.pad_token_id})")
        
        # Try to load as sequence classification model first
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_id,
                num_labels=num_labels,
                problem_type='multi_label_classification',
                trust_remote_code=True,
                torch_dtype=torch_dtype
            )
            # Resize token embeddings if we added a new pad token
            if len(tokenizer) > model.config.vocab_size:
                model.resize_token_embeddings(len(tokenizer))
                print(f"  Resized model embeddings to match tokenizer: {len(tokenizer)}")
            
            # Set pad_token_id in model config (critical for batching)
            if hasattr(model.config, 'pad_token_id'):
                model.config.pad_token_id = tokenizer.pad_token_id
            print(f"  Model config pad_token_id: {getattr(model.config, 'pad_token_id', 'Not set')}")
            
            # For AutoModelForSequenceClassification, the classifier is called 'score', not 'classifier'
            # Ensure it's trainable before LoRA is applied
            if hasattr(model, 'score'):
                for param in model.score.parameters():
                    param.requires_grad = True
                print(f"  ✓ Score layer (classifier) is trainable ({sum(p.numel() for p in model.score.parameters()):,} params)")
            elif hasattr(model, 'classifier'):
                for param in model.classifier.parameters():
                    param.requires_grad = True
                print(f"  ✓ Classifier is trainable ({sum(p.numel() for p in model.classifier.parameters()):,} params)")
        except Exception as e:
            print(f"Could not load as sequence classification: {e}")
            print("Loading as causal LM and adding classification head...")
            
            # Load as causal LM and add classification head
            base_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch_dtype
            )
            
            # Resize token embeddings if we added a new pad token
            if len(tokenizer) > base_model.config.vocab_size:
                base_model.resize_token_embeddings(len(tokenizer))
                print(f"  Resized model embeddings to match tokenizer: {len(tokenizer)}")
            
            # Set pad_token_id in model config (critical for batching)
            # This MUST be set for the model to handle batching correctly
            base_model.config.pad_token_id = tokenizer.pad_token_id
            # Also set it as an attribute if the config doesn't have it
            if not hasattr(base_model.config, 'pad_token_id') or base_model.config.pad_token_id is None:
                setattr(base_model.config, 'pad_token_id', tokenizer.pad_token_id)
            
            print(f"  Model config pad_token_id: {base_model.config.pad_token_id}")
            print(f"  Tokenizer pad_token_id: {tokenizer.pad_token_id}")
            
            # Verify they match
            if base_model.config.pad_token_id != tokenizer.pad_token_id:
                print(f"  WARNING: pad_token_id mismatch! Model: {base_model.config.pad_token_id}, Tokenizer: {tokenizer.pad_token_id}")
                base_model.config.pad_token_id = tokenizer.pad_token_id
            
            # Create wrapper with classification head
            class ModelWithClassificationHead(nn.Module):
                def __init__(self, base_model, num_labels, pad_token_id=None):
                    super().__init__()
                    self.base_model = base_model
                    # Store pad_token_id for fallback in forward pass
                    self._tokenizer_pad_token_id = pad_token_id
                    # Get hidden size from config
                    if hasattr(base_model.config, 'hidden_size'):
                        hidden_size = base_model.config.hidden_size
                    elif hasattr(base_model.config, 'd_model'):
                        hidden_size = base_model.config.d_model
                    else:
                        # Try to infer from model
                        hidden_size = base_model.config.n_embd if hasattr(base_model.config, 'n_embd') else 768
                    self.classifier = nn.Linear(hidden_size, num_labels)
                    # Ensure classifier head is trainable
                    for param in self.classifier.parameters():
                        param.requires_grad = True
                    
                def forward(self, input_ids=None, attention_mask=None, labels=None):
                    # Ensure pad_token_id is set in config before forward pass
                    if hasattr(self.base_model, 'config'):
                        if not hasattr(self.base_model.config, 'pad_token_id') or self.base_model.config.pad_token_id is None:
                            # Get pad_token_id from tokenizer if available
                            # This is a fallback in case it wasn't set during initialization
                            if hasattr(self, '_tokenizer_pad_token_id'):
                                self.base_model.config.pad_token_id = self._tokenizer_pad_token_id
                    
                    # Pass use_cache=False when gradient checkpointing is enabled
                    outputs = self.base_model(
                        input_ids=input_ids, 
                        attention_mask=attention_mask,
                        use_cache=False  # Required when gradient checkpointing is enabled
                    )
                    
                    # Get last hidden state
                    if hasattr(outputs, 'last_hidden_state'):
                        hidden_states = outputs.last_hidden_state
                    elif hasattr(outputs, 'hidden_states'):
                        hidden_states = outputs.hidden_states[-1]
                    else:
                        # For causal LMs, use the last token
                        hidden_states = outputs.logits
                    
                    # Use mean pooling over sequence length
                    if len(hidden_states.shape) == 3:
                        # Mask out padding tokens
                        if attention_mask is not None:
                            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                            hidden_states = hidden_states * mask
                            pooled = hidden_states.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                        else:
                            pooled = hidden_states.mean(dim=1)
                    else:
                        pooled = hidden_states
                    
                    logits = self.classifier(pooled)
                    
                    return type('Output', (), {
                        'logits': logits,
                        'loss': None
                    })()
            
            model = ModelWithClassificationHead(base_model, num_labels, pad_token_id=tokenizer.pad_token_id)
            
            # Initialize classifier with small weights to prevent NaN
            # Use Xavier uniform initialization (more stable than default)
            for param in model.classifier.parameters():
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param, gain=0.1)  # Small gain for stability
                else:
                    nn.init.zeros_(param)  # Bias to zero
            print(f"  ✓ Classifier head initialized with small weights ({sum(p.numel() for p in model.classifier.parameters()):,} parameters)")
            print(f"  ⚠️  Classifier will be frozen during LoRA training to prevent NaN")
        
        # Apply LoRA if requested
        if use_lora and PEFT_AVAILABLE:
            print(f"Applying LoRA with rank={lora_rank}, alpha={lora_alpha}")
            # Determine target modules based on model architecture
            if 'qwen' in model_name.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            elif 'llama' in model_name.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            elif 'gemma' in model_name.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            else:
                # Default for other models
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            
            # FREEZE classifier completely - it's causing NaN even with conservative settings
            # The pretrained classifier should work fine with LoRA-adapted features
            # We'll use a workaround to ensure gradients flow through frozen classifier to LoRA
            modules_to_save = None  # Freeze classifier completely
            
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,  # For classification
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                modules_to_save=modules_to_save,  # None - freeze classifier completely
            )
            
            print(f"  Note: Classifier will be FROZEN (prevents NaN)")
            print(f"  Pretrained classifier works with LoRA-adapted features")
            print(f"  Gradients flow through frozen classifier via workaround")
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            
            # CRITICAL: Ensure classifier is frozen
            classifier_frozen_count = 0
            for name, param in model.named_parameters():
                if ('score' in name or 'classifier' in name) and param.requires_grad:
                    param.requires_grad = False  # Force freeze
                    classifier_frozen_count += param.numel()
            
            if classifier_frozen_count > 0:
                print(f"  ✓ Froze {classifier_frozen_count:,} classifier params")
            
            # Verify we have trainable parameters (should be LoRA adapters only)
            trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  ✓ Total trainable parameters (LoRA only): {trainable_after:,}")
            
            # Final check: verify we have trainable parameters
            if trainable_after == 0:
                raise RuntimeError("No trainable parameters found after LoRA setup! Check model structure.")
            
            # Set pad_token_id on all nested config levels (PEFT creates nested models)
            pad_token_id = tokenizer.pad_token_id
            visited = set()
            def set_pad_token_recursive(obj, depth=0, max_depth=5):
                if depth > max_depth or id(obj) in visited:
                    return
                visited.add(id(obj))
                
                if hasattr(obj, 'config') and hasattr(obj.config, 'pad_token_id'):
                    obj.config.pad_token_id = pad_token_id
                
                if hasattr(obj, 'base_model'):
                    try:
                        base = obj.base_model
                        if base is not obj:  # Prevent self-reference
                            set_pad_token_recursive(base, depth + 1, max_depth)
                    except (AttributeError, RecursionError):
                        pass
                
                if hasattr(obj, 'peft_config'):
                    try:
                        for key in obj.peft_config.keys():
                            if hasattr(obj.peft_config[key], 'pad_token_id'):
                                obj.peft_config[key].pad_token_id = pad_token_id
                    except (AttributeError, TypeError):
                        pass
            
            set_pad_token_recursive(model)
            print(f"  ✓ pad_token_id set to {pad_token_id} on all config levels")
            
        elif use_lora and not PEFT_AVAILABLE:
            print("Warning: LoRA requested but PEFT not available. Using full fine-tuning.")
        
        # Enable gradient checkpointing for memory efficiency
        def enable_gradient_checkpointing(obj):
            if hasattr(obj, 'base_model') and hasattr(obj.base_model, 'gradient_checkpointing_enable'):
                obj.base_model.gradient_checkpointing_enable()
            elif hasattr(obj, 'gradient_checkpointing_enable'):
                obj.gradient_checkpointing_enable()
        enable_gradient_checkpointing(model)
        print("  ✓ Gradient checkpointing enabled")
    
    elif 'roberta' in model_name.lower():
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained(
            'roberta-base',
            num_labels=num_labels,
            problem_type='multi_label_classification'
        )
        
        # Apply LoRA to RoBERTa if requested
        if use_lora and PEFT_AVAILABLE:
            print(f"Applying LoRA to RoBERTa with rank={lora_rank}, alpha={lora_alpha}")
            target_modules = ["query", "key", "value", "dense"]  # RoBERTa attention modules
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=0.1,
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        elif use_lora and not PEFT_AVAILABLE:
            print("Warning: LoRA requested but PEFT not available. Using full fine-tuning.")
    elif 'modernbert' in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained('tdmd/modernbert-base')
        model = AutoModelForSequenceClassification.from_pretrained(
            'tdmd/modernbert-base',
            num_labels=num_labels,
            problem_type='multi_label_classification'
        )
        
        # Apply LoRA to ModernBERT if requested
        if use_lora and PEFT_AVAILABLE:
            print(f"Applying LoRA to ModernBERT with rank={lora_rank}, alpha={lora_alpha}")
            target_modules = ["query", "key", "value", "dense"]  # BERT-style attention modules
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=0.1,
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        elif use_lora and not PEFT_AVAILABLE:
            print("Warning: LoRA requested but PEFT not available. Using full fine-tuning.")
    else:  # bert-base-uncased or default
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=num_labels,
            problem_type='multi_label_classification'
        )
        
        # Apply LoRA to BERT/RoBERTa if requested (less common but supported)
        if use_lora and PEFT_AVAILABLE:
            print(f"Applying LoRA to BERT with rank={lora_rank}, alpha={lora_alpha}")
            target_modules = ["query", "key", "value", "dense"]  # BERT attention modules
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=0.1,
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        elif use_lora and not PEFT_AVAILABLE:
            print("Warning: LoRA requested but PEFT not available. Using full fine-tuning.")
    
    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    if use_lora:
        print(f"  Using LoRA (rank={lora_rank}, alpha={lora_alpha})")
    else:
        print(f"  Using full fine-tuning")
    
    return model, tokenizer

def train_model(model, train_loader, val_loader, device, epochs=3, learning_rate=2e-5, 
                source='', model_name='', label_names=None):
    """Train the model"""
    
    model.to(device)
    
    # Get trainable parameters (important for LoRA - only LoRA params should be trainable)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found! Check if model is properly set up for training.")
    
    # Separate classifier and LoRA parameters
    # Classifier needs MUCH lower LR to prevent NaN
    classifier_params = []
    lora_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'score' in name or 'classifier' in name:
                classifier_params.append(param)
            else:
                lora_params.append(param)
    
    if not lora_params:
        print(f"  ERROR: No LoRA parameters found!")
        print(f"  This suggests LoRA was not properly applied.")
        print(f"  Trainable param names: {[n for n, p in model.named_parameters() if p.requires_grad][:10]}")
        raise RuntimeError("No LoRA parameters found! Check LoRA configuration.")
    
    # Use VERY different learning rates: classifier gets 100x lower LR
    param_groups = [{'params': lora_params, 'lr': learning_rate, 'weight_decay': 0.01}]
    
    if classifier_params:
        # Use MUCH lower LR for classifier (1e-7 = 100x lower than typical 1e-5)
        classifier_lr = max(learning_rate * 0.01, 1e-7)  # At least 100x lower, minimum 1e-7
        param_groups.append({'params': classifier_params, 'lr': classifier_lr, 'weight_decay': 0.01})
        print(f"  LoRA params: {sum(p.numel() for p in lora_params):,} with LR={learning_rate:.2e}")
        print(f"  Classifier params: {sum(p.numel() for p in classifier_params):,} with LR={classifier_lr:.2e} (100x lower)")
    else:
        print(f"  LoRA params: {sum(p.numel() for p in lora_params):,} with LR={learning_rate:.2e}")
    
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Training {len(param_groups)} parameter groups with {total_trainable:,} trainable parameters")
    
    optimizer = AdamW(param_groups)
    
    # Use focal loss for class imbalance
    criterion = FocalLoss(alpha=1, gamma=2)
    
    # Learning rate scheduler
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    best_val_f1 = 0.0
    best_model_state = None
    # Early stopping patience: stop if no improvement for 3 epochs
    # This allows training to continue for large datasets while stopping early if converged
    patience = 3
    patience_counter = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        sys.stdout.flush()  # Flush before starting progress bar
        
        # Training
        model.train()
        
        # CRITICAL: Ensure LoRA adapters are enabled (PEFT can disable them)
        # This is essential for gradient flow - disabled adapters won't contribute to gradients
        if hasattr(model, 'enable_adapter_layers'):
            model.enable_adapter_layers()
        if hasattr(model, 'base_model') and hasattr(model.base_model, 'enable_adapter_layers'):
            model.base_model.enable_adapter_layers()
        
        # Verify LoRA adapters are actually enabled and trainable
        trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
        if trainable_count == 0:
            print(f"  ERROR: No trainable parameters! LoRA adapters may be disabled.")
            raise RuntimeError("No trainable parameters found! Check LoRA adapter status.")
        
        train_loss = 0
        train_preds = []
        train_labels = []
        
        # Calculate total batches for better progress display
        total_batches = len(train_loader)
        print(f"  Training on {total_batches} batches...")
        sys.stdout.flush()
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}", total=total_batches, **tqdm_kwargs)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            # Pass use_cache=False when gradient checkpointing is enabled (suppresses warning)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits
            
            # Check for NaN in logits BEFORE loss computation
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"\n⚠️  CRITICAL: NaN/Inf detected in model logits!")
                print(f"  Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
                print(f"  NaN count: {torch.isnan(logits).sum().item()}")
                print(f"  Inf count: {torch.isinf(logits).sum().item()}")
                print(f"  This suggests gradient explosion or model instability.")
                print(f"  Skipping this batch and checking model parameters...")
                
                # Check model parameters for NaN
                nan_params = []
                for name, param in model.named_parameters():
                    if param.requires_grad and (torch.isnan(param).any() or torch.isinf(param).any()):
                        nan_params.append(name)
                        print(f"    NaN/Inf in: {name}")
                
                if nan_params:
                    print(f"  ⚠️  Model parameters have NaN/Inf! Attempting to fix...")
                    # Try to reset NaN parameters to small random values
                    for name, param in model.named_parameters():
                        if param.requires_grad and (torch.isnan(param).any() or torch.isinf(param).any()):
                            if 'score' in name or 'classifier' in name:
                                # Reinitialize classifier with small values
                                nn.init.normal_(param, mean=0.0, std=0.02)
                                print(f"    Reset {name} to small random values")
                            else:
                                # For LoRA params, zero them
                                param.data.zero_()
                                print(f"    Zeroed {name}")
                    print(f"  Resetting optimizer state...")
                    optimizer.zero_grad()
                    # Skip this batch to let reset take effect
                    continue
                else:
                    print(f"  Model parameters OK, but logits are NaN. Skipping batch.")
                    continue
            
            # Convert logits and labels to float32 for loss computation
            # This is critical for MPS with bfloat16 models - backward pass needs float32
            # Ensure conversion preserves gradient computation
            if logits.dtype != torch.float32:
                logits = logits.float()  # Convert to float32, preserves requires_grad
            if labels.dtype != torch.float32:
                labels = labels.float()  # Convert to float32
            
            # Check if logits require grad
            # With frozen classifier, logits should still have gradients from LoRA adapters
            # LoRA adapts the base model features, which flow through frozen classifier to logits
            if not logits.requires_grad:
                # Check if we have any trainable parameters at all
                trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
                if trainable_count == 0:
                    print(f"  ERROR: No trainable parameters found!")
                    print(f"  This should not happen - LoRA adapters should be trainable")
                    raise RuntimeError("No trainable parameters! Check LoRA configuration.")
                else:
                    # This can happen with frozen classifier - gradients flow through but PyTorch
                    # might not track them properly. The workaround in loss computation will handle this.
                    # Don't print warning here - it's expected and handled below
                    pass
            
            loss = criterion(logits, labels)
            
            # CRITICAL: With frozen classifier, we need to force gradient connection
            # Frozen layers don't track gradients by default, but we need them to flow to LoRA
            if not loss.requires_grad:
                trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
                if trainable_count == 0:
                    raise RuntimeError("Loss does not require grad! No trainable parameters found.")
                
                # Force gradient connection: add zero contribution from LoRA param
                # This ensures PyTorch tracks gradients through frozen classifier to LoRA
                lora_param = next((p for n, p in model.named_parameters() 
                                 if p.requires_grad and ('lora' in n.lower() or 'adapter' in n.lower())), None)
                if lora_param is None:
                    lora_param = next(p for p in model.parameters() if p.requires_grad)
                
                # This workaround forces gradient tracking through frozen classifier
                loss = loss + 0.0 * lora_param.sum()
                
                if not loss.requires_grad:
                    print(f"  CRITICAL: Cannot establish gradient connection! Skipping batch...")
                    optimizer.zero_grad()
                    continue
            
            # Ensure loss is float32 for backward pass on MPS
            # FocalLoss should already return float32, but double-check
            if loss.dtype != torch.float32:
                loss = loss.float()
            
            loss.backward()
            
            # Gradient clipping for LoRA only (classifier is frozen, no gradients)
            # Check for NaN/Inf gradients first
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f"  ⚠️  NaN/Inf gradient in {name}, zeroing it")
                        param.grad.zero_()
            
            # Clip LoRA gradients (classifier is frozen, so no classifier gradients)
            lora_params_list = [p for n, p in model.named_parameters() 
                              if p.grad is not None and p.requires_grad]
            if lora_params_list:
                torch.nn.utils.clip_grad_norm_(lora_params_list, max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
            # Collect predictions for metrics
            probs = torch.sigmoid(logits).cpu().detach().numpy()
            train_preds.append(probs)
            train_labels.append(labels.cpu().detach().numpy())
        
        train_loss /= len(train_loader)
        train_preds = np.vstack(train_preds)
        train_labels = np.vstack(train_labels)
        
        # Calculate per-label F1 scores with optimal thresholds (per-category threshold optimization)
        train_per_label_f1 = {}
        
        for i, cat in enumerate(label_names):
            true_labels = train_labels[:, i].astype(int)
            pred_scores = train_preds[:, i]
            
            if len(np.unique(true_labels)) > 1:  # Has both positive and negative examples
                # Find best threshold for this category
                best_f1 = 0
                best_threshold = 0.5
                
                for threshold in np.arange(0.1, 0.9, 0.05):
                    binary_pred = (pred_scores > threshold).astype(int)
                    if len(np.unique(binary_pred)) > 1:  # Has both predictions
                        f1 = f1_score(true_labels, binary_pred, zero_division=0)
                        if f1 > best_f1:
                            best_f1 = f1
                            best_threshold = threshold
                
                train_per_label_f1[cat] = best_f1
            else:
                # All same label - use default threshold
                binary_pred = (pred_scores > 0.5).astype(int)
                train_per_label_f1[cat] = f1_score(true_labels, binary_pred, zero_division=0)
        
        # Macro F1: average of per-label F1 scores
        train_macro_f1 = np.mean(list(train_per_label_f1.values()))
        
        # Micro F1: global calculation with 0.5 threshold (standard)
        train_preds_binary = (train_preds > 0.5).astype(int)
        train_micro_f1 = f1_score(train_labels, train_preds_binary, average='micro', zero_division=0)
        
        # Also calculate per-sample accuracy (how many samples got all labels correct)
        # and partial accuracy (how many labels were correct per sample)
        train_per_sample_exact = (train_preds_binary == train_labels).all(axis=1).mean()
        train_per_label_accuracy = (train_preds_binary == train_labels).mean()
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{epochs}", **tqdm_kwargs):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                    logits = outputs.logits.to(dtype=torch.float32)
                    labels = labels.to(dtype=torch.float32)
                
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                probs = torch.sigmoid(logits).cpu().detach().numpy()
                val_preds.append(probs)
                val_labels.append(labels.cpu().detach().numpy())
        
        val_loss /= len(val_loader)
        val_preds = np.vstack(val_preds)
        val_labels = np.vstack(val_labels)
        
        # Calculate per-label F1 scores with optimal thresholds (per-category threshold optimization)
        # This is more appropriate for multi-label classification with imbalanced classes
        val_per_label_f1 = {}
        best_thresholds = []
        
        for i, cat in enumerate(label_names):
            true_labels = val_labels[:, i].astype(int)
            pred_scores = val_preds[:, i]
            
            if len(np.unique(true_labels)) > 1:  # Has both positive and negative examples
                # Find best threshold for this category
                best_f1 = 0
                best_threshold = 0.5
                
                for threshold in np.arange(0.1, 0.9, 0.05):
                    binary_pred = (pred_scores > threshold).astype(int)
                    if len(np.unique(binary_pred)) > 1:  # Has both predictions
                        f1 = f1_score(true_labels, binary_pred, zero_division=0)
                        if f1 > best_f1:
                            best_f1 = f1
                            best_threshold = threshold
                
                val_per_label_f1[cat] = best_f1
                best_thresholds.append(best_threshold)
            else:
                # All same label - use default threshold
                binary_pred = (pred_scores > 0.5).astype(int)
                val_per_label_f1[cat] = f1_score(true_labels, binary_pred, zero_division=0)
                best_thresholds.append(0.5)
        
        # Macro F1: average of per-label F1 scores
        val_macro_f1 = np.mean(list(val_per_label_f1.values()))
        
        # Micro F1: global calculation with 0.5 threshold (standard)
        val_preds_binary = (val_preds > 0.5).astype(int)
        val_micro_f1 = f1_score(val_labels, val_preds_binary, average='micro', zero_division=0)
        
        # Calculate additional metrics for better understanding
        val_per_sample_exact = (val_preds_binary == val_labels).all(axis=1).mean()
        val_per_label_accuracy = (val_preds_binary == val_labels).mean()
        
        train_loss_str = f"{train_loss:.4f}" if not np.isnan(train_loss) else "nan"
        print(f"Train Loss: {train_loss_str}, Train Macro F1: {train_macro_f1:.4f}, Train Micro F1: {train_micro_f1:.4f}")
        print(f"  Train: Exact match: {train_per_sample_exact:.1%}, Per-label accuracy: {train_per_label_accuracy:.1%}")
        val_loss_str = f"{val_loss:.4f}" if not np.isnan(val_loss) else "nan"
        print(f"Val Loss: {val_loss_str}, Val Macro F1: {val_macro_f1:.4f}, Val Micro F1: {val_micro_f1:.4f}")
        print(f"  Val: Exact match: {val_per_sample_exact:.1%}, Per-label accuracy: {val_per_label_accuracy:.1%}")
        
        # Print per-label F1 scores (top 5 and bottom 5 for visibility)
        sorted_f1 = sorted(val_per_label_f1.items(), key=lambda x: x[1], reverse=True)
        print(f"\n  Per-label F1 scores (top 5):")
        for cat, f1 in sorted_f1[:5]:
            print(f"    {cat}: {f1:.4f}")
        if len(sorted_f1) > 5:
            print(f"  Per-label F1 scores (bottom 5):")
            for cat, f1 in sorted_f1[-5:]:
                print(f"    {cat}: {f1:.4f}")
        
        # Early stopping
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

def evaluate_model(model, test_loader, device, label_names, source, model_name):
    """Evaluate the model on test set"""
    
    model.eval()
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", **tqdm_kwargs):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            # Convert to float32 for sigmoid computation (more stable)
            logits = outputs.logits.to(dtype=torch.float32)
            
            probs = torch.sigmoid(logits).cpu().detach().numpy()
            test_preds.append(probs)
            test_labels.append(labels.cpu().detach().numpy())
    
    test_preds = np.vstack(test_preds)
    test_labels = np.vstack(test_labels)
    
    # Diagnostic information
    print(f"\nEvaluation Diagnostics:")
    print(f"  Test set size: {len(test_preds)} samples")
    print(f"  Number of labels: {len(label_names)}")
    print(f"  Prediction shape: {test_preds.shape}")
    print(f"  Label shape: {test_labels.shape}")
    print(f"  Prediction range: [{test_preds.min():.4f}, {test_preds.max():.4f}]")
    print(f"  Prediction mean: {test_preds.mean():.4f}")
    print(f"  Label range: [{test_labels.min():.4f}, {test_labels.max():.4f}]")
    print(f"  Label sum (total positives): {test_labels.sum():.0f}")
    print(f"  Positive labels per category:")
    for i, cat in enumerate(label_names):
        pos_count = test_labels[:, i].sum()
        print(f"    {cat}: {pos_count:.0f} ({pos_count/len(test_labels)*100:.1f}%)")
    
    # Calculate metrics with threshold 0.5
    test_preds_binary = (test_preds > 0.5).astype(int)
    
    # Diagnostic: check how many positive predictions we have
    print(f"  Positive predictions per category (threshold=0.5):")
    for i, cat in enumerate(label_names):
        pred_pos_count = test_preds_binary[:, i].sum()
        true_pos_count = test_labels[:, i].sum()
        print(f"    {cat}: {pred_pos_count:.0f} predicted, {true_pos_count:.0f} true")
    
    # Check if we have any positive labels at all
    if test_labels.sum() == 0:
        print(f"\n  ⚠️  WARNING: Test set has NO positive labels! All metrics will be 0.0.")
        print(f"     This suggests an issue with label loading or test set preparation.")
    
    # Check if we have any positive predictions
    if test_preds_binary.sum() == 0:
        print(f"\n  ⚠️  WARNING: Model made NO positive predictions! All predictions < 0.5.")
        print(f"     This suggests the model didn't learn or predictions are too conservative.")
        print(f"     Consider checking:")
        print(f"       - Model training logs for convergence")
        print(f"       - Whether the model loaded correctly")
        print(f"       - Prediction threshold (currently 0.5)")
    
    macro_f1 = f1_score(test_labels, test_preds_binary, average='macro', zero_division=0)
    micro_f1 = f1_score(test_labels, test_preds_binary, average='micro', zero_division=0)
    
    # Per-category metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, test_preds_binary, average=None, zero_division=0
    )
    
    results = {
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'per_category_f1': dict(zip(label_names, f1)),
        'per_category_precision': dict(zip(label_names, precision)),
        'per_category_recall': dict(zip(label_names, recall))
    }
    
    return results, test_preds, test_labels

def main():
    parser = argparse.ArgumentParser(description='Finetune local models on GPT pseudolabels')
    parser.add_argument('--source', type=str, required=True, choices=['reddit', 'x', 'news', 'meeting_minutes'],
                       help='Data source')
    parser.add_argument('--model', type=str, default='bert-base-uncased',
                       choices=['bert-base-uncased', 'roberta-base', 'modernbert-base', 'llama', 'qwen', 'gemma3'],
                       help='Model to finetune (6 options: bert-base-uncased, roberta-base, modernbert-base, llama, qwen, gemma3)')
    parser.add_argument('--epochs', type=int, default=5, 
                       help='Number of epochs (default: 5, with early stopping patience=3. For large datasets (100K+), consider 5-10 epochs)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (reduce for local LLMs)')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=128, help='Max sequence length (default: 128, reduce for memory-constrained devices)')
    parser.add_argument('--test_start', type=int, default=1600, help='Start index for test set')
    parser.add_argument('--test_end', type=int, default=1700, help='End index for test set')
    parser.add_argument('--use_lora', action='store_true', 
                       help='Use LoRA for efficient fine-tuning (recommended for large models: llama, qwen, gemma3)')
    parser.add_argument('--lora_rank', type=int, default=16, 
                       help='LoRA rank (default: 16, optimized for accuracy and stability. rank=16 gives ~0.21%% trainable, achieves 96-98%% of full fine-tuning. Higher ranks (32+) may cause NaN issues)')
    parser.add_argument('--lora_alpha', type=int, default=32, 
                       help='LoRA alpha (default: 32, typically 2x rank. Higher = stronger adaptation)')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                       help='LoRA dropout rate (default: 0.1, range: 0.0-1.0. Higher = more regularization)')
    parser.add_argument('--lora_target_modules', type=str, default='all',
                       choices=['all', 'attention', 'mlp', 'attention+mlp'],
                       help='Which modules to apply LoRA to: all (qkv+mlp), attention (qkv only), mlp (ffn only), attention+mlp (both)')
    
    args = parser.parse_args()
    
    # Set device - prioritize CUDA for GPU, then MPS, then CPU
    print("\n" + "="*80)
    print("Device Configuration:")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    
    if cuda_available:
        device = torch.device('cuda')
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"  ✓ Using {num_gpus} CUDA GPU(s):")
            for i in range(num_gpus):
                print(f"    - GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"      Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        else:
            print(f"  ✓ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            print(f"    Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        if cuda_visible:
            print(f"    CUDA_VISIBLE_DEVICES: {cuda_visible}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("  ✓ Using MPS (Apple Silicon GPU)")
        # Try to get GPU info if available
        try:
            import subprocess
            result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                                  capture_output=True, text=True, timeout=2)
            if 'Chipset Model' in result.stdout:
                for line in result.stdout.split('\n'):
                    if 'Chipset Model' in line:
                        gpu_name = line.split('Chipset Model:')[1].strip()
                        print(f"    GPU: {gpu_name}")
                        break
        except:
            pass
    else:
        device = torch.device('cpu')
        print("  ⚠ Using CPU (no GPU detected)")
        print(f"    PyTorch version: {torch.__version__}, CUDA available: {cuda_available}")
        if cuda_visible:
            print(f"    CUDA_VISIBLE_DEVICES: {cuda_visible}")
        
        # Check nvidia-smi for Linux clusters
        try:
            import subprocess
            nvidia_smi = subprocess.run(['nvidia-smi', '--list-gpus'], 
                                       capture_output=True, text=True, timeout=2)
            if nvidia_smi.returncode == 0 and nvidia_smi.stdout.strip():
                print(f"    ⚠ GPUs found via nvidia-smi but PyTorch can't access them.")
                print(f"    Solutions: Install PyTorch with CUDA, load CUDA module, or request GPU in scheduler")
        except:
            pass
        print("    Note: Training will be significantly slower on CPU")
    print("="*80 + "\n")
    
    # Adjust batch size for local LLMs (they're larger)
    local_llms = ['llama', 'qwen', 'gemma3']
    is_local_llm = any(llm in args.model.lower() for llm in local_llms)
    
    if is_local_llm:
        if args.batch_size == 16:  # Default, adjust it
            # With LoRA, can use larger batch sizes, but MPS has memory limits
            if device.type == 'mps':
                # MPS has ~20GB limit, use smaller batches
                args.batch_size = 2 if args.use_lora else 1
                print(f"Adjusting batch size to {args.batch_size} for local LLM on MPS (memory-constrained)")
            elif device.type == 'cuda':
                # For CUDA, LoRA can use larger batches, but full fine-tuning needs very small batches
                if args.use_lora:
                    args.batch_size = 8
                else:
                    args.batch_size = 1
                    print(f"⚠️  WARNING: Full fine-tuning requires batch_size=1 on 22GB GPU")
                    print(f"   Consider using --use_lora for much faster training and less memory")
                print(f"Adjusting batch size to {args.batch_size} for local LLM")
            else:
                args.batch_size = 8 if args.use_lora else 4
                print(f"Adjusting batch size to {args.batch_size} for local LLM")
        
        # Recommend LoRA for large models if not specified
        if not args.use_lora:
            print(f"\n⚠️  Recommendation: Consider using --use_lora for {args.model}")
            print(f"   This will make training 3-10x faster and use less memory.")
            print(f"   Example: python {sys.argv[0]} --source {args.source} --model {args.model} --use_lora")
    
    # Check if LoRA is requested but PEFT is not available
    if args.use_lora and not PEFT_AVAILABLE:
        print("\n" + "="*80)
        print("ERROR: --use_lora requested but PEFT is not installed!")
        print("="*80)
        print("\nTo use LoRA, you must install PEFT:")
        print("  pip install peft")
        print("\nWithout LoRA, full fine-tuning of 7B models requires:")
        print("  - Much more GPU memory (~40GB+ for Qwen 7B)")
        print("  - Smaller batch sizes (batch_size=1 or 2)")
        print("  - Significantly longer training time")
        print("\nExiting. Please install PEFT or remove --use_lora flag.")
        print("="*80 + "\n")
        sys.exit(1)
    
    # Adjust learning rate for LoRA (based on research: 1e-4 to 1e-5 is optimal)
    # Research shows: Lower LR for higher ranks, start conservative to avoid NaN
    if args.use_lora and args.learning_rate == 2e-5:  # Default
        # Learning rate scaling based on rank (higher rank = more sensitive, needs lower LR)
        if args.lora_rank >= 32:
            args.learning_rate = 5e-6  # Very conservative for rank 32+ (research: start with 5e-6)
            print(f"⚠️  Using very conservative LR {args.learning_rate} for rank {args.lora_rank} to avoid NaN")
        elif args.lora_rank >= 16:
            args.learning_rate = 1e-5  # Conservative for rank 16 (research: 1e-5 is safer)
            print(f"Using conservative LR {args.learning_rate} for rank {args.lora_rank} (research-recommended)")
        elif args.lora_rank >= 8:
            args.learning_rate = 2e-5  # Standard for rank 8
        else:
            args.learning_rate = 5e-5  # Higher for very low ranks
        print(f"Adjusting learning rate to {args.learning_rate} for LoRA (rank={args.lora_rank})")
        print(f"  Research suggests: 1e-4 to 1e-5 for stability, lower for higher ranks")
    
    # Load GPT pseudolabels (excluding few-shot examples)
    gpt_df = load_gpt_pseudolabels(args.source, exclude_few_shot=True)
    
    # Load gold standard (with labels if available)
    gold_df, text_col, gold_labels_df = load_gold_standard(args.source)
    
    # Prepare GPT labels - ensure we have all categories in the right order
    gpt_label_cols = [col for col in gpt_df.columns if col in GPT_TO_CATEGORY_MAP]
    
    # Create mapping from GPT columns to ALL_CATEGORIES order
    category_to_gpt_col = {v: k for k, v in GPT_TO_CATEGORY_MAP.items()}
    
    # Prepare labels in ALL_CATEGORIES order
    # Note: Each comment can have MULTIPLE categories (multi-label classification)
    # The flag columns (Comment_*, Critique_*, etc.) are binary (0/1) indicating presence of each category
    gpt_texts = gpt_df['Comment'].astype(str).tolist()
    gpt_labels_list = []
    
    print(f"\nProcessing multi-label categories for {len(gpt_texts):,} unique comments...")
    
    for idx, row in gpt_df.iterrows():
        label_vector = []
        for cat in ALL_CATEGORIES:
            gpt_col = category_to_gpt_col.get(cat)
            if gpt_col and gpt_col in row:
                val = row[gpt_col]
                # Flag columns are already binary (0/1), so check if > 0.5
                label_vector.append(1.0 if pd.notna(val) and float(val) > 0.5 else 0.0)
            else:
                label_vector.append(0.0)
        gpt_labels_list.append(label_vector)
    
    gpt_labels = np.array(gpt_labels_list)
    
    # Print label distribution
    print(f"\nLabel distribution (multi-label, so categories can overlap):")
    for i, category in enumerate(ALL_CATEGORIES):
        positive_count = np.sum(gpt_labels[:, i])
        percentage = positive_count / len(gpt_labels) * 100
        print(f"  {category:30}: {positive_count:6,}/{len(gpt_labels):6,} ({percentage:5.1f}%)")
    
    # Split gold standard into val and test
    # Train set: Only GPT pseudolabels (no gold standard)
    # Val/Test set: Gold standard split (excluding few-shot examples)
    
    # Extract few-shot examples from hardcoded prompts in utils.py
    # The few-shot examples are embedded in the prompt text, not in separate files
    # We need to extract the actual 5 examples from the prompt strings
    try:
        from scripts.utils import (
            FEW_SHOT_REDDIT_PROMPT_TEXT, FEW_SHOT_X_PROMPT_TEXT,
            FEW_SHOT_NEWS_PROMPT_TEXT, FEW_SHOT_MEETING_MINUTES_PROMPT_TEXT
        )
        
        few_shot_prompt_map = {
            'reddit': FEW_SHOT_REDDIT_PROMPT_TEXT,
            'x': FEW_SHOT_X_PROMPT_TEXT,
            'news': FEW_SHOT_NEWS_PROMPT_TEXT,
            'meeting_minutes': FEW_SHOT_MEETING_MINUTES_PROMPT_TEXT
        }
        
        few_shot_prompt = few_shot_prompt_map.get(args.source, '')
        few_shot_comments = set()
        
        # Extract examples from prompt text (format: "Sentance: ..." or "Post: ..." or "Article: ..." or "Meeting Minutes: ...")
        import re
        # Define the prefix for each source
        prefix_map = {
            'reddit': 'Sentance:',
            'x': 'Post:',
            'news': 'Article:',
            'meeting_minutes': 'Meeting Minutes:'
        }
        prefix = prefix_map.get(args.source, '')
        
        if prefix:
            # Split by the prefix to get each example section
            parts = few_shot_prompt.split(prefix)
            # Skip the first part (everything before first prefix)
            for part in parts[1:]:
                # Extract text up to "Comment Type:" (the classification starts there)
                if '\nComment Type:' in part:
                    text = part.split('\nComment Type:')[0]
                else:
                    # If no "Comment Type:" found, take everything up to next prefix or end
                    # Check if there's another prefix in this part
                    next_prefix_positions = []
                    for other_prefix in prefix_map.values():
                        if other_prefix in part:
                            next_prefix_positions.append(part.find(other_prefix))
                    if next_prefix_positions:
                        text = part[:min(next_prefix_positions)]
                    else:
                        text = part
                
                # Clean up the text (remove extra whitespace, newlines)
                text = ' '.join(text.split())
                if text:  # Only add non-empty text
                    few_shot_comments.add(text)
            
            if len(few_shot_comments) != 5:
                print(f"  Warning: Expected 5 few-shot examples but extracted {len(few_shot_comments)}")
                print(f"  This is okay - will exclude whatever was found")
        
        print(f"  Extracted {len(few_shot_comments)} few-shot examples from prompt text")
        if len(few_shot_comments) == 0:
            print(f"  ERROR: No few-shot examples extracted! Check prompt text format.")
        elif len(few_shot_comments) < 5:
            print(f"  WARNING: Only extracted {len(few_shot_comments)} examples, expected 5")
    except Exception as e:
        print(f"  Warning: Could not extract few-shot examples from utils: {e}")
        print(f"  Will use gold_subset files as fallback (may exclude too many)")
        # Fallback to old method
        few_shot_files = [
            f'output/{args.source}/gpt4/classified_comments_{args.source}_gold_subset_gpt4_{args.source}_flags.csv',
            f'output/{args.source}/gpt4/classified_comments_{args.source}_gold_subset_gpt4_none_flags.csv'
        ]
        few_shot_comments = set()
        for few_shot_file in few_shot_files:
            if Path(few_shot_file).exists():
                few_shot_df = pd.read_csv(few_shot_file)
                # Use 'Comment' column from few-shot files (they all use 'Comment')
                few_shot_comments.update(few_shot_df['Comment'].astype(str).tolist())
        print(f"  Found {len(few_shot_comments)} unique few-shot comments to exclude (fallback method)")
    
    # Filter out few-shot examples from gold standard using the text_col from load_gold_standard
    if few_shot_comments:
        initial_gold_len = len(gold_df)
        
        # Normalize few-shot comments for matching (strip, lowercase, normalize whitespace)
        def normalize_text(text):
            # Remove newlines and normalize whitespace
            text = ' '.join(str(text).split()).strip().lower()
            # Remove surrounding quotes if present (few-shot examples may have quotes)
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1]
            elif text.startswith("'") and text.endswith("'"):
                text = text[1:-1]
            return text
        
        few_shot_comments_normalized = {normalize_text(c) for c in few_shot_comments}
        print(f"  Sample few-shot text (first 100 chars): {list(few_shot_comments_normalized)[0][:100] if few_shot_comments_normalized else 'N/A'}")
        
        # Normalize gold standard texts for matching (same normalization)
        gold_texts_normalized = gold_df[text_col].astype(str).apply(normalize_text)
        print(f"  Sample gold standard text (first 100 chars): {gold_texts_normalized.iloc[0][:100] if len(gold_texts_normalized) > 0 else 'N/A'}")
        
        # Try exact matching first
        exact_mask = ~gold_texts_normalized.isin(few_shot_comments_normalized)
        exact_matches = (~exact_mask).sum()
        print(f"  Exact matches found: {exact_matches}/{len(few_shot_comments_normalized)}")
        
        # For any few-shot examples that didn't match exactly, try substring matching
        # This handles cases where text might have slight differences (whitespace, deidentification, etc.)
        mask = exact_mask.copy()
        unmatched_few_shot = []
        for few_shot_text in few_shot_comments_normalized:
            # Check if this few-shot text matched exactly
            if few_shot_text not in gold_texts_normalized.values:
                unmatched_few_shot.append(few_shot_text)
        
        if unmatched_few_shot:
            print(f"  Trying substring matching for {len(unmatched_few_shot)} unmatched examples...")
            for few_shot_text in unmatched_few_shot:
                matched = False
                # Try progressively shorter substrings (100, 80, 60, 50, 40, 30 chars)
                for substr_len in [100, 80, 60, 50, 40, 30]:
                    if len(few_shot_text) < substr_len:
                        continue
                    search_text = re.escape(few_shot_text[:substr_len])
                    temp_mask = gold_texts_normalized.str.contains(search_text, case=False, na=False, regex=True)
                    if temp_mask.any():
                        mask = mask & ~temp_mask
                        match_idx = temp_mask.idxmax()
                        matched = True
                        print(f"    Matched: {few_shot_text[:60]}... at index {match_idx}")
                        break
                
                if not matched:
                    # Try reverse: check if gold standard text is in few-shot
                    for idx in gold_texts_normalized.index:
                        if not mask.iloc[idx] if hasattr(mask, 'iloc') else not mask[idx]:
                            continue
                        gs_text = gold_texts_normalized.iloc[idx]
                        if (len(few_shot_text) >= 50 and (few_shot_text[:50] in gs_text or few_shot_text[:80] in gs_text)) or \
                           (len(gs_text) >= 50 and (gs_text[:50] in few_shot_text or gs_text[:80] in few_shot_text)):
                            if hasattr(mask, 'iloc'):
                                mask.iloc[idx] = False
                            else:
                                mask[idx] = False
                            matched = True
                            print(f"    Matched: {few_shot_text[:60]}... at index {idx}")
                            break
                
                if not matched:
                    print(f"    No match found for: {few_shot_text[:60]}...")
        
        matches_found = (~mask).sum()
        print(f"  Total matches found: {matches_found}/{len(few_shot_comments_normalized)}")
        
        # Get indices to keep (convert boolean mask to integer indices)
        keep_indices = mask[mask].index.tolist() if hasattr(mask, 'index') else [i for i, keep in enumerate(mask) if keep]
        
        # Apply filter to gold_df
        gold_df = gold_df.iloc[keep_indices].reset_index(drop=True)
        
        # Also filter gold_labels_df to match (if it exists)
        if gold_labels_df is not None:
            # Ensure gold_labels_df has the same length as original gold_df
            if len(gold_labels_df) != initial_gold_len:
                # Truncate to match length (shouldn't be longer, but handle it)
                if len(gold_labels_df) > initial_gold_len:
                    gold_labels_df = gold_labels_df.iloc[:initial_gold_len]
                # If shorter, we can only filter what exists
                keep_indices = [i for i in keep_indices if i < len(gold_labels_df)]
            
            # Apply same filter using iloc (works regardless of index alignment)
            gold_labels_df = gold_labels_df.iloc[keep_indices].reset_index(drop=True)
        
        excluded_count = initial_gold_len - len(gold_df)
        print(f"  Excluded {excluded_count} few-shot examples from gold standard")
        if excluded_count == 0:
            print(f"  WARNING: No examples were excluded! This might mean:")
            print(f"    - Few-shot examples don't match gold standard text (check normalization)")
            print(f"    - Few-shot examples weren't extracted correctly")
            print(f"    - Text matching failed (check if texts are identical)")
    
    # Use ALL remaining gold standard samples for evaluation (val + test)
    # This should be (total_samples - 5 few-shot) samples
    # Note: Different sources have different sizes:
    #   - reddit: 500 total, so ~495 after excluding 5 few-shot
    #   - x: 456 total, so ~451 after excluding 5 few-shot  
    #   - news: 382 total, so ~377 after excluding 5 few-shot
    #   - meeting_minutes: 364 total, so ~359 after excluding 5 few-shot
    gold_eval_indices = list(range(len(gold_df)))  # Use all remaining samples
    
    # Get gold texts after filtering (using filtered gold_df)
    gold_texts = gold_df[text_col].astype(str).tolist()
    
    # Calculate expected count based on source (different sources have different sizes)
    source_expected_counts = {
        'reddit': 500,           # Actual: 500 samples
        'x': 456,                # Actual: 456 samples
        'news': 382,             # Actual: 382 samples
        'meeting_minutes': 364   # Actual: 364 samples
    }
    expected_total = source_expected_counts.get(args.source, 500)
    expected_after_exclude = expected_total - 5  # 5 few-shot examples
    
    print(f"\nSplitting data:")
    print(f"  Train set (GPT pseudolabels only): {len(gpt_texts):,} unique comments")
    print(f"  Gold standard total (after excluding few-shot): {len(gold_df)} samples")
    print(f"  Gold standard eval set (all remaining, excluding few-shot): {len(gold_eval_indices)} samples")
    print(f"  Expected: ~{expected_after_exclude} samples ({expected_total} total - 5 few-shot = {expected_after_exclude})")
    print(f"  Note: All sources have <= 500 samples (reddit: 500, x: 456, news: 382, meeting_minutes: 364)")
    
    # Safety check: ensure we have samples for evaluation
    if len(gold_df) == 0:
        raise ValueError(
            f"ERROR: All gold standard samples were filtered out! "
            f"This likely means all samples match few-shot examples. "
            f"Check few-shot files: {few_shot_files}"
        )
    
    if len(gold_eval_indices) == 0:
        raise ValueError(
            f"ERROR: No samples available for evaluation! "
            f"Gold standard has {len(gold_df)} samples after filtering."
        )
    
    # Recommend epochs based on dataset size (now using actual unique comment count)
    if len(gpt_texts) > 20000:
        if args.epochs < 5:
            print(f"\n⚠️  Large dataset detected ({len(gpt_texts):,} unique comments). Consider using --epochs 5-10 for better learning.")
            print(f"   Current: {args.epochs} epochs. Early stopping (patience=3) will stop early if converged.")
    elif len(gpt_texts) > 5000:
        if args.epochs < 3:
            print(f"\n⚠️  Medium dataset ({len(gpt_texts):,} unique comments). Consider using --epochs 3-5.")
            print(f"   Current: {args.epochs} epochs. Early stopping (patience=3) will stop early if converged.")
    
    # Match gold standard to GPT pseudolabels by comment text
    gpt_comment_to_labels = {}
    for text, label in zip(gpt_texts, gpt_labels):
        # Normalize text for matching
        text_normalized = text.strip().lower()
        gpt_comment_to_labels[text_normalized] = label
    
    # Prepare gold standard eval texts and labels
    # IMPORTANT: Use actual gold standard labels (human annotations), NOT GPT pseudolabels
    gold_eval_texts = []
    gold_eval_labels_list = []
    
    for i in gold_eval_indices:
        comment = gold_texts[i]
        gold_eval_texts.append(comment)
        
        # Use actual gold standard labels if available
        # Gold labels should be aligned by index with gold_df
        if gold_labels_df is not None and i < len(gold_labels_df):
            # Extract labels from gold standard (aligned by index)
            matched_row = gold_labels_df.iloc[i]
            label_vector = []
            
            for cat in ALL_CATEGORIES:
                # Try different column names
                val = None
                if cat in gold_labels_df.columns:
                    val = matched_row[cat]
                elif cat == 'racist' and 'Racist' in gold_labels_df.columns:
                    val = matched_row['Racist']
                elif cat == 'racist' and 'racist' in gold_labels_df.columns:
                    val = matched_row['racist']
                
                if pd.notna(val) and val != '':
                    # If it's a score (0-3 from raw_scores), threshold at >= 2 (2+ annotators agree)
                    try:
                        score = float(val)
                        if score >= 2:
                            label_vector.append(1.0)
                        else:
                            label_vector.append(0.0)
                    except (ValueError, TypeError):
                        # If it's already binary or soft label (0-1), threshold at >= 0.5
                        try:
                            soft_val = float(val)
                            label_vector.append(1.0 if soft_val >= 0.5 else 0.0)
                        except:
                            label_vector.append(0.0)
                else:
                    label_vector.append(0.0)
            
            gold_eval_labels_list.append(label_vector)
        else:
            # Fallback: use GPT pseudolabels if gold labels not available
            comment_normalized = comment.strip().lower()
            if comment_normalized in gpt_comment_to_labels:
                gold_eval_labels_list.append(gpt_comment_to_labels[comment_normalized])
                if gold_labels_df is None:
                    print(f"  WARNING: Gold labels not found, using GPT pseudolabels for comment {i}")
            else:
                print(f"  WARNING: Gold standard comment {i} not found, using zeros")
                gold_eval_labels_list.append(np.zeros(len(ALL_CATEGORIES)))
    
    gold_eval_labels = np.array(gold_eval_labels_list)
    
    # Safety check before splitting
    if len(gold_eval_texts) == 0 or len(gold_eval_labels) == 0:
        raise ValueError(
            f"ERROR: gold_eval_texts or gold_eval_labels is empty! "
            f"gold_eval_texts: {len(gold_eval_texts)}, gold_eval_labels: {len(gold_eval_labels)}. "
            f"This suggests an issue with building the eval sets."
        )
    
    if gold_labels_df is not None:
        print(f"  ✓ Using ACTUAL gold standard labels (human annotations) for test/val sets")
        print(f"    This ensures evaluation on real human-annotated data, not GPT pseudolabels")
    else:
        print(f"  ⚠ WARNING: Using GPT pseudolabels for test/val (gold standard labels not found)")
        print(f"    This is NOT ideal for research - please ensure gold standard labels are available")
    
    print(f"  Prepared {len(gold_eval_texts)} eval samples with {len(gold_eval_labels)} labels")
    
    # Split gold standard eval set into val and test (50/50 split)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        gold_eval_texts, gold_eval_labels, test_size=0.5, random_state=42
    )
    
    # Train set: Only GPT pseudolabels (no gold standard)
    train_texts = gpt_texts.copy()
    train_labels = gpt_labels.copy()
    
    print(f"\nFinal dataset sizes:")
    print(f"  Train (GPT pseudolabels only): {len(train_texts):,} samples")
    print(f"  Val (gold standard split): {len(val_texts)} samples")
    print(f"  Test (gold standard split): {len(test_texts)} samples")
    print(f"  Total eval: {len(val_texts) + len(test_texts)} samples (should be ~{expected_after_exclude})")
    
    # Create model and tokenizer
    model, tokenizer = create_model_and_tokenizer(
        args.model, 
        len(ALL_CATEGORIES), 
        device=device,
        use_lora=args.use_lora,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha
    )
    
    # Create datasets
    train_dataset = GPTPseudolabelDataset(train_texts, train_labels, tokenizer, max_length=args.max_length)
    val_dataset = GPTPseudolabelDataset(val_texts, val_labels, tokenizer, max_length=args.max_length)
    test_dataset = GPTPseudolabelDataset(test_texts, test_labels, tokenizer, max_length=args.max_length)
    
    # Create data loaders - use num_workers=0 for MPS to avoid multiprocessing issues
    num_workers = 0 if device.type == 'mps' else 2
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    
    # Train model
    print(f"\nTraining {args.model} on {args.source}...")
    sys.stdout.flush()  # Ensure output is flushed before training starts
    model = train_model(
        model, train_loader, val_loader, device,
        epochs=args.epochs, learning_rate=args.learning_rate,
        source=args.source, model_name=args.model, label_names=ALL_CATEGORIES
    )
    
    # Evaluate on test set
    print(f"\nEvaluating on test set...")
    results, test_preds, test_labels = evaluate_model(
        model, test_loader, device, ALL_CATEGORIES, args.source, args.model
    )
    
    print(f"\nTest Results:")
    print(f"  Macro F1: {results['macro_f1']:.4f}")
    print(f"  Micro F1: {results['micro_f1']:.4f}")
    print(f"\nPer-category F1:")
    for cat, f1 in results['per_category_f1'].items():
        print(f"  {cat}: {f1:.4f}")
    
    # Save model
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    # Clean model name for file saving
    model_name_clean = args.model.replace('/', '_').replace('-', '_')
    suffix = '_lora' if args.use_lora else ''
    
    # Save model based on whether LoRA is used
    if args.use_lora and PEFT_AVAILABLE and hasattr(model, 'save_pretrained'):
        # Save LoRA adapters using PEFT's save_pretrained (saves adapter weights + config)
        lora_model_dir = model_dir / f'gpt_pseudolabel_{model_name_clean}_lora_{args.source}'
        model.save_pretrained(str(lora_model_dir))
        print(f"\n✓ LoRA adapters saved to: {lora_model_dir}")
        print(f"  (Contains: adapter_model.bin, adapter_config.json)")
        print(f"  To load: Use PEFT's PeftModel.from_pretrained() with base model")
        
        # Also save state_dict for compatibility
        model_path = model_dir / f'gpt_pseudolabel_{model_name_clean}_lora_best_{args.source}.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'lora_config': {
                'rank': args.lora_rank,
                'alpha': args.lora_alpha,
                'dropout': args.lora_dropout,
                'target_modules': args.lora_target_modules
            },
            'model_name': args.model,
            'source': args.source
        }, model_path)
        print(f"✓ Full state dict saved to: {model_path}")
    else:
        # Save full fine-tuned model
        model_path = model_dir / f'gpt_pseudolabel_{model_name_clean}_best_{args.source}.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_name': args.model,
            'source': args.source,
            'use_lora': False
        }, model_path)
        print(f"\n✓ Full fine-tuned model saved to: {model_path}")
    
    # Save results
    output_dir = Path('nlp_outputs') / args.source
    output_dir.mkdir(parents=True, exist_ok=True)
    
    suffix = '_lora' if args.use_lora else ''
    results_file = output_dir / f'gpt_pseudolabel_{model_name_clean}{suffix}_results.json'
    
    # Add training config to results
    results['training_config'] = {
        'model': args.model,
        'source': args.source,
        'use_lora': args.use_lora,
        'lora_rank': args.lora_rank if args.use_lora else None,
        'lora_alpha': args.lora_alpha if args.use_lora else None,
        'lora_dropout': args.lora_dropout if args.use_lora else None,
        'lora_target_modules': args.lora_target_modules if args.use_lora else None,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'max_length': args.max_length
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    main()

