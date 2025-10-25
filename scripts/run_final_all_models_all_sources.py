#!/usr/bin/env python3
"""
Run Final Classifier for All Models Across All Sources

Trains the final classifier for each combination of:
- models: bert-base-uncased, roberta-base, modernbert-base
- sources: reddit, x, news, meeting_minutes

Usage:
    # Transformers (BERT, RoBERTa, ModernBERT) with and without SMOTE across all sources
    python scripts/run_final_all_models_all_sources.py [--epochs 3] [--batch_size 16] \
        [--learning_rate 2e-5] [--max_length 256]

    # Also include sklearn baselines (LogReg/SVM/RF) without SMOTE across all sources
    python scripts/run_final_all_models_all_sources.py --also_sklearn

Notes:
- This script calls scripts/bert_final_classifier.py in train mode.
"""

import argparse
import os
import subprocess
import time
from datetime import datetime


def _metrics_path(source: str, model: str, use_smote: bool) -> str:
    suffix = 'smote' if use_smote else 'original'
    model_tag = model.replace('-', '_')
    return f"nlp_outputs/{source}/{model_tag}_{suffix}_metrics.json"

def _csv_path(source: str, model: str, use_smote: bool) -> str:
    suffix = 'smote' if use_smote else 'original'
    model_tag = model.replace('-', '_')
    return f"nlp_outputs/{source}/{model_tag}_{suffix}.csv"


def run_training(source: str, model: str, args: argparse.Namespace, use_smote: bool) -> bool:
    print(f"\n{'=' * 80}")
    print(f"Training model={model} source={source} smote={use_smote}")
    print(f"{'=' * 80}")

    cmd = [
        'python3', 'scripts/bert_final_classifier.py',
        '--source', source,
        '--mode', 'train',
        '--model', model,
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--learning_rate', str(args.learning_rate),
        '--max_length', str(args.max_length),
    ]
    if use_smote:
        cmd.append('--use_smote')

    start = time.time()
    try:
        subprocess.run(cmd, check=True)
        dur = time.time() - start
        print(f"✅ Completed model={model} source={source} smote={use_smote} in {dur/60:.1f} min")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed model={model} source={source} smote={use_smote}")
        print(f"Return code: {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Run final classifier for all models and sources')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--max_length', type=int, default=256)
    # Always runs BOTH with and without SMOTE for transformers
    parser.add_argument('--also_sklearn', action='store_true', help='Also run sklearn baselines without SMOTE')
    parser.add_argument('--also', action='store_true', help='Alias for --also_sklearn')
    args = parser.parse_args()

    # Normalize alias
    if args.also and not args.also_sklearn:
        args.also_sklearn = True

    models = ['bert-base-uncased', 'roberta-base', 'modernbert-base']
    sources = ['reddit', 'x', 'news', 'meeting_minutes']

    print(f"Starting runs at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models: {', '.join(models)}")
    print(f"Sources: {', '.join(sources)}")
    print(f"Hyperparams: epochs={args.epochs} batch_size={args.batch_size} lr={args.learning_rate} max_len={args.max_length}")

    total_start = time.time()
    results = {}

    for model in models:
        for source in sources:
            # First, non-SMOTE
            key_no = f"{model}:{source}:no_smote"
            metrics_no = _metrics_path(source, model, use_smote=False)
            csv_no = _csv_path(source, model, use_smote=False)
            if os.path.exists(metrics_no) or os.path.exists(csv_no):
                print(f"Skipping (exists): {metrics_no if os.path.exists(metrics_no) else csv_no}")
                results[key_no] = True
                # Backfill combined CSVs if per-model outputs exist but combined rows missing
                try:
                    import pandas as pd
                    combined_path = 'nlp_outputs/all_results.csv'
                    transformers_path = 'nlp_outputs/all_transformer_results.csv'
                    overall_path = 'nlp_outputs/summary_overall_results.csv'
                    need_header = not os.path.exists(combined_path)
                    # If combined exists, check whether rows for this (model, source, smote=False) are present
                    if not need_header:
                        df_all = pd.read_csv(combined_path)
                        present = ((df_all.get('model') == model) & (df_all.get('source') == source) & (df_all.get('smote') == False)).any()
                    else:
                        present = False
                    if not present and os.path.exists(csv_no):
                        df = pd.read_csv(csv_no)
                        # Try to enrich with metrics JSON (macro/micro and splits)
                        try:
                            import json as _json
                            metrics_path = _metrics_path(source, model, use_smote=False)
                            metrics = None
                            if os.path.exists(metrics_path):
                                with open(metrics_path, 'r') as fh:
                                    metrics = _json.load(fh)
                        except Exception:
                            metrics = None

                        # Add minimal context columns
                        df['source'] = source
                        df['model'] = model
                        df['smote'] = False
                        # Optional enrichments
                        df['macro_f1'] = metrics.get('macro_f1') if metrics else None
                        df['micro_f1'] = metrics.get('micro_f1') if metrics else None
                        df['train'] = metrics.get('train_size') if metrics else None
                        df['val'] = metrics.get('val_size') if metrics else None
                        df['test'] = metrics.get('test_size') if metrics else None
                        if 'synthetic' not in df.columns:
                            # per-category CSV may have synthetic_added
                            df['synthetic'] = df['synthetic_added'] if 'synthetic_added' in df.columns else 0
                        # Append to combined
                        df.to_csv(combined_path, mode='a', header=need_header, index=False)
                        print(f"Backfilled combined CSV from {csv_no}")

                        # Backfill all_transformer_results.csv
                        try:
                            need_header_t = not os.path.exists(transformers_path)
                            # Conform headers
                            df_t = df.copy()
                            # Keep transformer schema columns
                            keep_t = ['category','source','model','smote','train','val','test','synthetic','macro_f1','micro_f1','subset_accuracy','hamming_loss','precision','recall','f1','accuracy','roc_auc','average_precision']
                            df_t = df_t[[c for c in keep_t if c in df_t.columns]]
                            df_t.to_csv(transformers_path, mode='a', header=need_header_t, index=False)
                            print(f"Backfilled transformer combined CSV from {csv_no}")
                        except Exception as exc:
                            print(f"Warning: transformer backfill failed: {exc}")

                        # Backfill overall summary
                        try:
                            need_header_o = not os.path.exists(overall_path)
                            overall_row = [{
                                'source': source,
                                'model': model,
                                'smote': False,
                                'macro_f1': metrics.get('macro_f1') if metrics else None,
                                'micro_f1': metrics.get('micro_f1') if metrics else None,
                                'subset_accuracy': metrics.get('subset_accuracy') if metrics else None,
                                'hamming_loss': metrics.get('hamming_loss') if metrics else None,
                                'train': metrics.get('train_size') if metrics else None,
                                'val': metrics.get('val_size') if metrics else None,
                                'test': metrics.get('test_size') if metrics else None,
                                'total_synthetic': metrics.get('total_synthetic_samples') if metrics else None
                            }]
                            pd.DataFrame(overall_row).to_csv(overall_path, mode='a', header=need_header_o, index=False)
                            print(f"Backfilled overall summary from {metrics_path if metrics else 'metrics unavailable'}")
                        except Exception as exc:
                            print(f"Warning: overall backfill failed: {exc}")
                except Exception as exc:
                    print(f"Warning: backfill failed for {model} {source} no_smote: {exc}")
            else:
                ok = run_training(source, model, args, use_smote=False)
                results[key_no] = ok

            # Then, SMOTE
            key_yes = f"{model}:{source}:smote"
            metrics_yes = _metrics_path(source, model, use_smote=True)
            csv_yes = _csv_path(source, model, use_smote=True)
            if os.path.exists(metrics_yes) or os.path.exists(csv_yes):
                print(f"Skipping (exists): {metrics_yes if os.path.exists(metrics_yes) else csv_yes}")
                results[key_yes] = True
                # Backfill combined CSVs if per-model outputs exist but combined rows missing
                try:
                    import pandas as pd
                    combined_path = 'nlp_outputs/all_results.csv'
                    transformers_path = 'nlp_outputs/all_transformer_results.csv'
                    overall_path = 'nlp_outputs/summary_overall_results.csv'
                    need_header = not os.path.exists(combined_path)
                    if not need_header:
                        df_all = pd.read_csv(combined_path)
                        present = ((df_all.get('model') == model) & (df_all.get('source') == source) & (df_all.get('smote') == True)).any()
                    else:
                        present = False
                    if not present and os.path.exists(csv_yes):
                        df = pd.read_csv(csv_yes)
                        # Try to enrich with metrics JSON
                        try:
                            import json as _json
                            metrics_path = _metrics_path(source, model, use_smote=True)
                            metrics = None
                            if os.path.exists(metrics_path):
                                with open(metrics_path, 'r') as fh:
                                    metrics = _json.load(fh)
                        except Exception:
                            metrics = None

                        df['source'] = source
                        df['model'] = model
                        df['smote'] = True
                        df['macro_f1'] = metrics.get('macro_f1') if metrics else None
                        df['micro_f1'] = metrics.get('micro_f1') if metrics else None
                        df['train'] = metrics.get('train_size') if metrics else None
                        df['val'] = metrics.get('val_size') if metrics else None
                        df['test'] = metrics.get('test_size') if metrics else None
                        if 'synthetic' not in df.columns:
                            df['synthetic'] = df['synthetic_added'] if 'synthetic_added' in df.columns else 0
                        df.to_csv(combined_path, mode='a', header=need_header, index=False)
                        print(f"Backfilled combined CSV from {csv_yes}")

                        # Backfill all_transformer_results.csv
                        try:
                            need_header_t = not os.path.exists(transformers_path)
                            df_t = df.copy()
                            keep_t = ['category','source','model','smote','train','val','test','synthetic','macro_f1','micro_f1','subset_accuracy','hamming_loss','precision','recall','f1','accuracy','roc_auc','average_precision']
                            df_t = df_t[[c for c in keep_t if c in df_t.columns]]
                            df_t.to_csv(transformers_path, mode='a', header=need_header_t, index=False)
                            print(f"Backfilled transformer combined CSV from {csv_yes}")
                        except Exception as exc:
                            print(f"Warning: transformer backfill failed: {exc}")

                        # Backfill overall summary
                        try:
                            need_header_o = not os.path.exists(overall_path)
                            overall_row = [{
                                'source': source,
                                'model': model,
                                'smote': True,
                                'macro_f1': metrics.get('macro_f1') if metrics else None,
                                'micro_f1': metrics.get('micro_f1') if metrics else None,
                                'subset_accuracy': metrics.get('subset_accuracy') if metrics else None,
                                'hamming_loss': metrics.get('hamming_loss') if metrics else None,
                                'train': metrics.get('train_size') if metrics else None,
                                'val': metrics.get('val_size') if metrics else None,
                                'test': metrics.get('test_size') if metrics else None,
                                'total_synthetic': metrics.get('total_synthetic_samples') if metrics else None
                            }]
                            pd.DataFrame(overall_row).to_csv(overall_path, mode='a', header=need_header_o, index=False)
                            print(f"Backfilled overall summary from {metrics_path if metrics else 'metrics unavailable'}")
                        except Exception as exc:
                            print(f"Warning: overall backfill failed: {exc}")
                except Exception as exc:
                    print(f"Warning: backfill failed for {model} {source} smote: {exc}")
            else:
                ok = run_training(source, model, args, use_smote=True)
                results[key_yes] = ok

    if args.also_sklearn:
        print("\nRunning sklearn baselines without SMOTE across all sources...")
        for source in sources:
            key = f"sklearn:{source}"
            cmd = [
                'python3', 'scripts/final_benchmark.py',
                '--source', source,
                '--mode', 'train',
                '--skip_transformers',
                '--sklearn_disable_smote'
            ]
            try:
                subprocess.run(cmd, check=True)
                results[key] = True
            except subprocess.CalledProcessError:
                results[key] = False

    total_time_min = (time.time() - total_start) / 60.0
    print(f"\n{'=' * 80}")
    print("RUN SUMMARY")
    print(f"{'=' * 80}")
    success = sum(1 for ok in results.values() if ok)
    total = len(results)
    print(f"Completed: {success}/{total} ({success/total*100:.1f}%) in {total_time_min:.1f} min")
    for key, ok in results.items():
        print(f"  {key}: {'✅' if ok else '❌'}")

    # Check for missing combinations in overall summary and rerun if needed
    try:
        import pandas as pd
        overall_path = 'nlp_outputs/summary_overall_results.csv'
        if os.path.exists(overall_path):
            df_overall = pd.read_csv(overall_path)
            expected_combinations = []
            for model in models:
                for source in sources:
                    for smote in [False, True]:
                        expected_combinations.append((model, source, smote))
            
            missing = []
            for model, source, smote in expected_combinations:
                present = ((df_overall.get('model') == model) & 
                          (df_overall.get('source') == source) & 
                          (df_overall.get('smote') == smote)).any()
                if not present:
                    missing.append((model, source, smote))
            
            if missing:
                print(f"\nFound {len(missing)} missing combinations in overall summary. Rerunning...")
                for model, source, smote in missing:
                    print(f"Rerunning: {model} {source} smote={smote}")
                    ok = run_training(source, model, args, use_smote=smote)
                    if ok:
                        print(f"✅ Completed rerun: {model} {source} smote={smote}")
                    else:
                        print(f"❌ Failed rerun: {model} {source} smote={smote}")
        else:
            print("No overall summary found - all combinations will be run fresh")
    except Exception as exc:
        print(f"Warning: missing combination check failed: {exc}")


if __name__ == '__main__':
    main()


