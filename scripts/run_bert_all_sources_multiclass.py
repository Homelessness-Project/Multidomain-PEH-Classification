#!/usr/bin/env python3
"""
Run BERT Multiclass Training for All Sources

This script trains BERT models for all sources (reddit, x, news, meeting_minutes)
using the comprehensive multiclass classifier.

Usage:
    python scripts/run_bert_all_sources_multiclass.py
"""

import subprocess
import sys
import os
import time
from datetime import datetime

def run_bert_training(source, dataset='gold_subset'):
    """Run BERT training for a specific source"""
    print(f"\n{'='*60}")
    print(f"Training BERT Multiclass for {source}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    cmd = [
        'python', 'scripts/bert_multiclass_classifier.py',
        '--source', source,
        '--dataset', dataset,
        '--mode', 'train',
        '--epochs', '5',
        '--batch_size', '16',
        '--learning_rate', '2e-5',
        '--max_length', '256'
    ]
    
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        end_time = time.time()
        duration = end_time - start_time
        print(f"✅ Successfully trained {source} in {duration:.1f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error training {source}: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def main():
    """Main function to run BERT training for all sources"""
    
    sources = ['reddit', 'x', 'news', 'meeting_minutes']
    dataset = 'gold_subset'  # Use gold subset for training
    
    print(f"Starting BERT Multiclass Training for All Sources")
    print(f"Dataset: {dataset}")
    print(f"Sources: {', '.join(sources)}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Track results
    results = {}
    total_start_time = time.time()
    
    for source in sources:
        print(f"\n{'='*80}")
        print(f"Processing {source.upper()}")
        print(f"{'='*80}")
        
        success = run_bert_training(source, dataset)
        results[source] = success
        
        if success:
            print(f"✅ {source} completed successfully")
        else:
            print(f"❌ {source} failed")
    
    # Summary
    total_duration = time.time() - total_start_time
    successful = sum(results.values())
    total = len(results)
    
    print(f"\n{'='*80}")
    print(f"TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"Total time: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    print(f"Successful: {successful}/{total}")
    print(f"Success rate: {successful/total*100:.1f}%")
    
    print(f"\nResults by source:")
    for source, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {source}: {status}")
    
    if successful == total:
        print(f"\n🎉 All models trained successfully!")
    else:
        print(f"\n⚠️  {total - successful} models failed to train")
    
    print(f"\nModels saved to: models/")
    print(f"Results saved to: output/*/bert/")

if __name__ == "__main__":
    main() 