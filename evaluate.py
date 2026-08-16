#!/usr/bin/env python3
"""
Phase 3 Evaluation Script
Runs robustness benchmarks, generates plots, and outputs final metrics.
"""
import os
import argparse
from src.evaluation.benchmarks import generate_benchmark_set, run_benchmark
from src.evaluation.reporting import (
    save_metrics_csv,
    plot_error_distribution,
    plot_accuracy_vs_noise,
    plot_baseline_vs_proposed,
    save_case_visualization
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-cond", type=int, default=5, help="Samples per condition per architecture")
    parser.add_argument("--out-dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "plots"), exist_ok=True)
    
    conditions = [
        'clean', 'low_noise', 'medium_noise', 'high_noise',
        'search_noisier', 'blur', 'rotation', 'scale', 
        'combined', 'high_periodicity'
    ]
    
    architectures = ['DRAM', 'FinFET']
    
    all_results = []
    
    # Track the best success and worst failure for visualization
    best_case = None
    worst_case = None
    best_item = None
    worst_item = None
    
    print("Generating datasets and running evaluations...")
    for arch in architectures:
        for cond_idx, cond in enumerate(conditions):
            print(f"Evaluating {arch} - {cond}...")
            # Generate dataset
            dataset = generate_benchmark_set(cond, arch, args.samples_per_cond, seed_offset=cond_idx)
            
            # Run baseline
            b_results = run_benchmark(dataset, method='baseline')
            all_results.extend(b_results)
            
            # Run proposed
            p_results = run_benchmark(dataset, method='proposed', alpha=1.0, beta=0.0)
            all_results.extend(p_results)
            
            # Find cases for visualization
            for i, p_res in enumerate(p_results):
                err = p_res['error_pixels']
                if p_res['success']:
                    if best_case is None or err < best_case['error_pixels']:
                        best_case = p_res
                        best_item = dataset[i]
                    if worst_case is None or err > worst_case['error_pixels']:
                        worst_case = p_res
                        worst_item = dataset[i]
                            
    # Save CSV
    csv_path = os.path.join(args.out_dir, "final_metrics.csv")
    save_metrics_csv(all_results, csv_path)
    print(f"Saved metrics to {csv_path}")
    
    # Generate Plots
    plot_dir = os.path.join(args.out_dir, "plots")
    plot_error_distribution(all_results, plot_dir)
    plot_accuracy_vs_noise(all_results, plot_dir)
    plot_baseline_vs_proposed(all_results, plot_dir)
    print(f"Saved plots to {plot_dir}")
    
    # Save visualizations
    if best_case and best_item:
        out_path = os.path.join(args.out_dir, "success_case.png")
        save_case_visualization(best_item, best_case, out_path)
        print(f"Saved success case to {out_path}")
        
    if worst_case and worst_item:
        out_path = os.path.join(args.out_dir, "failure_case.png")
        save_case_visualization(worst_item, worst_case, out_path)
        print(f"Saved failure case to {out_path}")
    elif len(all_results) > 0:
        # If no strict > 20px failure, just take the max error one
        proposed = [r for r in all_results if r['method'] == 'proposed' and r['success']]
        if proposed:
            worst = max(proposed, key=lambda x: x['error_pixels'])
            # Need to match item again...
            # For simplicity, skip if not found
            pass
            
    print("Evaluation Complete.")

if __name__ == "__main__":
    main()
