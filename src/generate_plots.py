#!/usr/bin/env python3
"""
Script pour visualiser les métriques des fichiers JSON de logs détaillés.
Crée un plot pour chaque métrique montrant son évolution en fonction des timesteps.

Usage:
    python src/plot_detailed_logs.py
    python src/plot_detailed_logs.py --log-file results/detailed_logs/experiment_metrics.json
    python src/plot_detailed_logs.py --log-dir results/detailed_logs --seed 0
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import os
import sys


def check_and_confirm_overwrite(output_dir, force=False, global_choice=None):
    """Vérifie si le dossier existe et demande confirmation pour écraser
    
    Args:
        output_dir: Le dossier de sortie à vérifier
        force: Si True, écrase automatiquement sans demander
        global_choice: Choix global ('overwrite_all', 'skip_all', ou None pour demander)
    
    Returns:
        tuple: (should_process, updated_global_choice)
    """
    if os.path.exists(output_dir):
        png_files = list(Path(output_dir).glob('*.png'))
        if png_files:
            if force:
                print(f"Overwriting {len(png_files)} existing plot(s) (--force enabled)")
                return True, global_choice
            if global_choice == 'overwrite_all':
                print(f"Overwriting {len(png_files)} existing plot(s) (apply to all)")
                return True, global_choice
            elif global_choice == 'skip_all':
                print(f"Skipping {len(png_files)} existing plot(s) (apply to all)")
                return False, global_choice
            
            print(f"\nWarning: Output directory already exists with {len(png_files)} plot(s)")
            print(f"Path: {output_dir}")
            print(f"Options:")
            print(f"     y  - Overwrite this directory")
            print(f"     n  - Skip this directory")
            print(f"     a  - Overwrite ALL conflicting directories")
            print(f"     s  - Skip ALL conflicting directories")
            
            while True:
                response = input("Your choice [y/n/a/s]: ").strip().lower()
                if response in ['y', 'yes']:
                    print("Overwriting this directory")
                    return True, global_choice
                elif response in ['n', 'no']:
                    print("Skipped")
                    return False, global_choice
                elif response == 'a':
                    print("Will overwrite ALL conflicting directories")
                    return True, 'overwrite_all'
                elif response == 's':
                    print("Will skip ALL conflicting directories")
                    return False, 'skip_all'
                else:
                    print("Invalid choice. Please enter y, n, a, or s")
    
    return True, global_choice


def load_json_log(json_path):
    """Charge un fichier JSON de log détaillé"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def extract_metrics_data(json_data):
    """Extrait les timesteps et métriques du fichier JSON"""
    metrics_dict = {}
    timesteps = []

    for t_str, metrics in json_data["metrics"].items():
        t = int(t_str)
        timesteps.append(t)

        for metric_name, value in metrics.items():
            if metric_name == "timestamp":
                continue
            
            if metric_name not in metrics_dict:
                metrics_dict[metric_name] = {"timesteps": [], "values": []}
            
            metrics_dict[metric_name]["timesteps"].append(t)
            metrics_dict[metric_name]["values"].append(value)

    for metric_name in metrics_dict:
        sorted_indices = np.argsort(metrics_dict[metric_name]["timesteps"])
        metrics_dict[metric_name]["timesteps"] = [
            metrics_dict[metric_name]["timesteps"][i] for i in sorted_indices
        ]
        metrics_dict[metric_name]["values"] = [
            metrics_dict[metric_name]["values"][i] for i in sorted_indices
        ]
    
    return metrics_dict, json_data["config"]


def plot_metric(metric_name, timesteps, values, config, output_dir):
    """Crée un plot pour une métrique donnée"""
    plt.figure(figsize=(12, 6))
    plt.plot(timesteps, values, linewidth=2, marker='o', markersize=4, alpha=0.7)
    plt.xlabel('Timesteps', fontsize=12, fontweight='bold')
    plt.ylabel(metric_name, fontsize=12, fontweight='bold')
    plt.title(f'{metric_name} - {config["algorithm"]} on {config["environment"]} (seed={config["seed"]})', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    safe_metric_name = metric_name.replace("/", "_").replace(" ", "_")
    output_file = os.path.join(output_dir, f"{safe_metric_name}.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_file}")


def plot_all_metrics(json_path, output_dir=None, force=False, global_choice=None):
    """Crée des plots pour toutes les métriques d'un fichier JSON"""
    print(f"\n Processing: {json_path}")
    json_data = load_json_log(json_path)
    metrics_dict, config = extract_metrics_data(json_data)
    if output_dir is None:
        json_path_obj = Path(json_path)
        base_dir = json_path_obj.parent / "seed_plots"
        output_dir = base_dir / f"{json_path_obj.stem}_plots"
    should_process, updated_global_choice = check_and_confirm_overwrite(output_dir, force, global_choice)
    if not should_process:
        return None, updated_global_choice
    
    os.makedirs(output_dir, exist_ok=True)
    print(f" Output directory: {output_dir}")
    print(f"\nConfiguration:")
    print(f"Algorithm: {config['algorithm']}")
    print(f"Environment: {config['environment']}")
    print(f"Seed: {config['seed']}")
    print(f"Perception: {config.get('perception_config', 'unknown')}")
    print(f"Started: {config['started_at']}")
    if 'finished_at' in config:
        print(f"Finished: {config['finished_at']}")
    print(f"\nGenerating plots for {len(metrics_dict)} metrics...")
    for metric_name, data in metrics_dict.items():
        plot_metric(metric_name, data["timesteps"], data["values"], config, output_dir)
    create_combined_plot(metrics_dict, config, output_dir)
    
    print(f"\n Done! Generated {len(metrics_dict) + 1} plots in {output_dir}")
    return output_dir, updated_global_choice


def create_combined_plot(metrics_dict, config, output_dir):
    """Crée un plot combiné avec les métriques principales"""
    priority_metrics = [
        "test_return_mean", "test_return_std", "return_mean", "return_std",
        "td_error_abs", "target_mean", "q_taken_mean",
        "advantage_mean", "pg_loss", "critic_loss", "pi_max",
        "agent_grad_norm", "critic_grad_norm", "grad_norm",
        "loss", "epsilon"
    ]
    available_metrics = [m for m in priority_metrics if m in metrics_dict]
    if len(available_metrics) < 6:
        other_important = ["ep_length_mean", "test_ep_length_mean"]
        available_metrics.extend([m for m in other_important if m in metrics_dict and m not in available_metrics])
    
    if not available_metrics:
        return
    n_metrics = len(available_metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'Training Metrics - {config["algorithm"]} on {config["environment"]} (seed={config["seed"]})', 
                 fontsize=16, fontweight='bold')
    
    for idx, metric_name in enumerate(available_metrics):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        data = metrics_dict[metric_name]
        ax.plot(data["timesteps"], data["values"], linewidth=2, marker='o', markersize=3)
        ax.set_xlabel('Timesteps', fontweight='bold')
        ax.set_ylabel(metric_name, fontweight='bold')
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    for idx in range(len(available_metrics), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, "combined_metrics.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved combined plot: {output_file}")


def find_json_logs(log_dir, seed=None):
    """Trouve tous les fichiers JSON de logs dans un répertoire"""
    log_dir = Path(log_dir)
    json_files = list(log_dir.glob("*_metrics.json"))
    
    if seed is not None:
        json_files = [f for f in json_files if f"seed{seed}" in f.name]
    
    return sorted(json_files)


def group_logs_by_experiment(json_files):
    """Groupe les fichiers JSON par (algorithme, environnement, t_max, perception_config)"""
    groups = {}
    
    for json_file in json_files:
        data = load_json_log(json_file)
        config = data["config"]
        key = (
            config["algorithm"], 
            config["environment"], 
            config.get("t_max", 200000),
            config.get("perception_config", "unknown")
        )
        
        if key not in groups:
            groups[key] = []
        
        groups[key].append({
            "path": json_file,
            "data": data,
            "seed": config["seed"]
        })
    
    return groups


def aggregate_metrics_across_seeds(experiments):
    """Agrège les métriques de plusieurs seeds"""
    all_metrics = {}
    
    for exp in experiments:
        metrics_dict, _ = extract_metrics_data(exp["data"])
        
        for metric_name, data in metrics_dict.items():
            if metric_name not in all_metrics:
                all_metrics[metric_name] = {}
            for t, value in zip(data["timesteps"], data["values"]):
                if t not in all_metrics[metric_name]:
                    all_metrics[metric_name][t] = []
                all_metrics[metric_name][t].append(value)
    aggregated = {}
    for metric_name, timestep_data in all_metrics.items():
        timesteps = sorted(timestep_data.keys())
        means = []
        stds = []
        
        for t in timesteps:
            values = timestep_data[t]
            means.append(np.mean(values))
            stds.append(np.std(values))
        
        aggregated[metric_name] = {
            "timesteps": timesteps,
            "mean": means,
            "std": stds,
            "n_seeds": len(experiments)
        }
    
    return aggregated


def plot_metric_with_std(metric_name, data, config, output_dir):
    """Crée un plot avec moyenne et écart-type"""
    timesteps = data["timesteps"]
    mean = np.array(data["mean"])
    std = np.array(data["std"])
    
    plt.figure(figsize=(12, 6))
    plt.plot(timesteps, mean, linewidth=2, label='Mean', color='blue')
    plt.fill_between(timesteps, mean - std, mean + std, alpha=0.3, color='blue', label='±1 std')
    
    plt.xlabel('Timesteps', fontsize=12, fontweight='bold')
    plt.ylabel(metric_name, fontsize=12, fontweight='bold')
    plt.title(f'{metric_name} - {config["algorithm"]} on {config["environment"]} ({data["n_seeds"]} seeds)', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    safe_metric_name = metric_name.replace("/", "_").replace(" ", "_")
    output_file = os.path.join(output_dir, f"{safe_metric_name}_aggregated.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def plot_aggregated_metrics(experiments, output_dir, force=False, global_choice=None):
    """Crée des plots agrégés pour un groupe d'expériences"""
    if not experiments:
        return None, global_choice
    config = experiments[0]["data"]["config"]
    seeds = [exp["seed"] for exp in experiments]
    
    print(f"\nAggregating {len(experiments)} experiments")
    print(f"Algorithm: {config['algorithm']}")
    print(f"Environment: {config['environment']}")
    print(f"T_max: {config.get('t_max', 200000)}")
    print(f"Perception: {config.get('perception_config', 'unknown')}")
    print(f"Seeds: {sorted(seeds)}")
    if output_dir is None:
        base_dir = Path("results/detailed_logs/aggregated_plots")
        output_dir = base_dir / (f"{config['algorithm']}_{config['environment']}_"
                   f"t{config.get('t_max', 200000)}_"
                   f"{config.get('perception_config', 'unknown')}")
        output_dir = base_dir
    should_process, updated_global_choice = check_and_confirm_overwrite(output_dir, force, global_choice)
    if not should_process:
        return None, updated_global_choice
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    aggregated = aggregate_metrics_across_seeds(experiments)
    print(f"\nGenerating aggregated plots for {len(aggregated)} metrics...")
    for metric_name, data in aggregated.items():
        plot_metric_with_std(metric_name, data, config, output_dir)
    create_combined_aggregated_plot(aggregated, config, output_dir)
    
    print(f"\nDone! Generated {len(aggregated) + 1} aggregated plots in {output_dir}")
    return output_dir, updated_global_choice


def create_combined_aggregated_plot(aggregated, config, output_dir):
    """Crée un plot combiné avec les métriques principales agrégées"""
    priority_metrics = [
        "test_return_mean", "test_return_std", "return_mean", "return_std",
        "td_error_abs", "target_mean", "q_taken_mean",
        "advantage_mean", "pg_loss", "critic_loss", "pi_max",
        "agent_grad_norm", "critic_grad_norm", "grad_norm",
        "loss", "epsilon"
    ]
    available_metrics = [m for m in priority_metrics if m in aggregated]
    if len(available_metrics) < 6:
        other_important = ["ep_length_mean", "test_ep_length_mean"]
        available_metrics.extend([m for m in other_important if m in aggregated and m not in available_metrics])
    
    if not available_metrics:
        return
    n_metrics = len(available_metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    n_seeds = aggregated[available_metrics[0]]["n_seeds"]
    fig.suptitle(f'Aggregated Training Metrics - {config["algorithm"]} on {config["environment"]} ({n_seeds} seeds)', 
                 fontsize=16, fontweight='bold')
    
    for idx, metric_name in enumerate(available_metrics):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        data = aggregated[metric_name]
        timesteps = data["timesteps"]
        mean = np.array(data["mean"])
        std = np.array(data["std"])
        
        ax.plot(timesteps, mean, linewidth=2, color='blue')
        ax.fill_between(timesteps, mean - std, mean + std, alpha=0.3, color='blue')
        ax.set_xlabel('Timesteps', fontweight='bold')
        ax.set_ylabel(metric_name, fontweight='bold')
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    for idx in range(len(available_metrics), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, "combined_metrics_aggregated.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved combined plot: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot metrics from JSON logs',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--log-dir', type=str, default='results/detailed_logs',
                        help='Directory containing JSON log files (default: results/detailed_logs)')
    parser.add_argument('--no-aggregate', action='store_true',
                        help='Skip aggregated plots generation')
    parser.add_argument('--only-aggregate', action='store_true',
                        help='Only generate aggregated plots (skip individual plots)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Base output directory for plots')
    parser.add_argument('--force', action='store_true',
                        help='Force overwrite without confirmation')
    
    args = parser.parse_args()
    if not os.path.exists(args.log_dir):
        print(f"Error: Directory not found: {args.log_dir}")
        return
    json_files = find_json_logs(args.log_dir, seed=None)
    
    if not json_files:
        print(f"No JSON log files found in {args.log_dir}")
        return
    
    print(f"{'='*80}")
    print(f"Found {len(json_files)} JSON log file(s) in {args.log_dir}")
    print(f"{'='*80}\n")
    groups = group_logs_by_experiment(json_files)
    
    print(f"Detected {len(groups)} unique experiment configuration(s):")
    for idx, ((algo, env, t_max, obs_level), exps) in enumerate(groups.items(), 1):
        seeds = sorted([exp["seed"] for exp in exps])
        print(f"   {idx}. {algo} on {env} (t_max={t_max}, obs={obs_level}) - {len(exps)} seed(s): {seeds}")
    print()
    if not args.only_aggregate:
        print(f"\n{'='*80}")
        print(f"Plots per seed")
        print(f"{'='*80}")
        
        global_choice = None
        for json_file in json_files:
            result, global_choice = plot_all_metrics(str(json_file), args.output_dir, args.force, global_choice)
            print()
    if not args.no_aggregate:
        print(f"\n{'='*80}")
        print(f"Aggregated plots")
        print(f"{'='*80}")
        for (algo, env, t_max, obs_level), experiments in groups.items():
            print(f"\n{'='*70}")
            print(f"Processing: {algo} on {env} (t_max={t_max}, obs={obs_level})")
            print(f"{'='*70}")
            result, global_choice = plot_aggregated_metrics(experiments, args.output_dir, args.force, global_choice)
    print(f"COMPLETE!")
    print(f"\nResults locations:")
    print(f"- Individual plots: results/detailed_logs/seed_plots/<experiment_name>_plots/")
    print(f"- Aggregated plots: results/detailed_logs/aggregated_plots/<algo>_<env>_t<steps>_<obs>/")
    print()


if __name__ == "__main__":
    main()
