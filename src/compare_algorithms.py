#!/usr/bin/env python3
"""
Comparer les performances de différents algorithmes sur différents environnements.

Usage:
    python src/plot_compare_algorithms.py
    python src/plot_compare_algorithms.py --env SimpleSpeakerListener-v0
    python src/plot_compare_algorithms.py --metric test_return_mean
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import os


def load_aggregated_data(aggregated_dir):
    """Charge toutes les données agrégées disponibles"""
    aggregated_dir = Path(aggregated_dir)
    
    if not aggregated_dir.exists():
        print(f"Aggregated plots directory not found: {aggregated_dir}")
        return {}
    
    experiments = {}
    for exp_dir in aggregated_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        parts = exp_dir.name.split('_')
        if len(parts) < 4:
            continue
        t_max_idx = None
        for i, part in enumerate(parts):
            if part.startswith('t') and len(part) > 1 and part[1:].isdigit():
                t_max_idx = i
                t_max = int(part[1:])
                break  
        if t_max_idx is None:
            continue
        obs_level = '_'.join(parts[t_max_idx+1:])
        env_idx = None
        for i in range(1, t_max_idx):
            if '-' in parts[i]:
                env_idx = i
                break
        if env_idx is None:
            continue
        algo = '_'.join(parts[:env_idx])
        env = '_'.join(parts[env_idx:t_max_idx])        
        if t_max is None:
            continue
        key = (algo, env, t_max, obs_level)
        
        if key not in experiments:
            experiments[key] = {
                'algo': algo,
                'env': env,
                't_max': t_max,
                'obs_level': obs_level,
                'dir': exp_dir
            }
    
    return experiments


def load_metrics_from_json_logs(log_dir, algo, env, t_max, obs_level):
    """Charge les métriques agrégées depuis les fichiers JSON de logs"""
    log_dir = Path(log_dir)
    json_files = list(log_dir.glob("*_metrics.json"))
    
    # Filtrer par algo, env, t_max, obs_level
    matching_files = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                config = data['config']
                
                if (config['algorithm'] == algo and 
                    config['environment'] == env and
                    config.get('t_max', 200000) == t_max and
                    config.get('perception_config', 'unknown') == obs_level):
                    matching_files.append((json_file, data))
        except:
            continue
    
    if not matching_files:
        return None
    all_metrics = {}
    
    for json_file, data in matching_files:
        for t_str, metrics in data['metrics'].items():
            t = int(t_str)
            
            for metric_name, value in metrics.items():
                if metric_name == 'timestamp':
                    continue
                
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = {}
                
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
            'timesteps': timesteps,
            'mean': means,
            'std': stds,
            'n_seeds': len(matching_files)
        }
    
    return aggregated


def plot_comparison(experiments_data, metric_name, output_dir, colors=None):
    """Crée un plot de comparaison pour une métrique donnée"""
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    plt.figure(figsize=(14, 8))
    
    color_idx = 0
    legend_labels = []
    
    for (algo, env, t_max, obs_level), data in sorted(experiments_data.items()):
        if metric_name not in data['metrics']:
            continue
        
        metric_data = data['metrics'][metric_name]
        timesteps = metric_data['timesteps']
        mean = np.array(metric_data['mean'])
        std = np.array(metric_data['std'])
        
        color = colors[color_idx % len(colors)]
        label = f"{algo} ({env})"
        plt.plot(timesteps, mean, linewidth=2.5, color=color, label=label, alpha=0.9)
        plt.fill_between(timesteps, mean - std, mean + std, alpha=0.2, color=color)
        
        legend_labels.append(label)
        color_idx += 1
    
    plt.xlabel('Timesteps', fontsize=14, fontweight='bold')
    plt.ylabel(metric_name, fontsize=14, fontweight='bold')
    plt.title(f'Algorithm Comparison - {metric_name}', fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    safe_metric_name = metric_name.replace("/", "_").replace(" ", "_")
    output_file = os.path.join(output_dir, f"comparison_{safe_metric_name}.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")


def plot_comparison_by_env(experiments_data, metric_name, output_dir, colors=None):
    """Crée des plots de comparaison séparés par environnement"""
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
    by_env = {}
    for (algo, env, t_max, obs_level), data in experiments_data.items():
        if env not in by_env:
            by_env[env] = {}
        by_env[env][(algo, env, t_max, obs_level)] = data
    for env, env_exps in by_env.items():
        plt.figure(figsize=(14, 8))
        
        color_idx = 0
        has_data = False
        
        for (algo, env_name, t_max, obs_level), data in sorted(env_exps.items()):
            if metric_name not in data['metrics']:
                continue
            
            has_data = True
            metric_data = data['metrics'][metric_name]
            timesteps = metric_data['timesteps']
            mean = np.array(metric_data['mean'])
            std = np.array(metric_data['std'])
            
            color = colors[color_idx % len(colors)]
            label = f"{algo} (t_max={t_max})"
            plt.plot(timesteps, mean, linewidth=2.5, color=color, label=label, alpha=0.9)
            plt.fill_between(timesteps, mean - std, mean + std, alpha=0.2, color=color)
            
            color_idx += 1
        
        if not has_data:
            plt.close()
            continue
        
        plt.xlabel('Timesteps', fontsize=14, fontweight='bold')
        plt.ylabel(metric_name, fontsize=14, fontweight='bold')
        plt.title(f'Algorithm Comparison on {env} - {metric_name}', fontsize=16, fontweight='bold')
        plt.legend(loc='best', fontsize=11, framealpha=0.9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        safe_metric_name = metric_name.replace("/", "_").replace(" ", "_")
        safe_env_name = env.replace("-", "_")
        output_file = os.path.join(output_dir, f"comparison_{safe_env_name}_{safe_metric_name}.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_file}")


def plot_comparison_by_algo(experiments_data, metric_name, output_dir, colors=None):
    """Crée des plots de comparaison séparés par algorithme"""
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))

    by_algo = {}
    for (algo, env, t_max, obs_level), data in experiments_data.items():
        if algo not in by_algo:
            by_algo[algo] = {}
        by_algo[algo][(algo, env, t_max, obs_level)] = data

    for algo, algo_exps in by_algo.items():
        plt.figure(figsize=(14, 8))
        
        color_idx = 0
        has_data = False
        
        for (algo_name, env, t_max, obs_level), data in sorted(algo_exps.items()):
            if metric_name not in data['metrics']:
                continue
            
            has_data = True
            metric_data = data['metrics'][metric_name]
            timesteps = metric_data['timesteps']
            mean = np.array(metric_data['mean'])
            std = np.array(metric_data['std'])
            
            color = colors[color_idx % len(colors)]

            label = f"{env} (t_max={t_max})"
            plt.plot(timesteps, mean, linewidth=2.5, color=color, label=label, alpha=0.9)
            plt.fill_between(timesteps, mean - std, mean + std, alpha=0.2, color=color)
            
            color_idx += 1
        
        if not has_data:
            plt.close()
            continue
        
        plt.xlabel('Timesteps', fontsize=14, fontweight='bold')
        plt.ylabel(metric_name, fontsize=14, fontweight='bold')
        plt.title(f'{algo} on Different Environments - {metric_name}', fontsize=16, fontweight='bold')
        plt.legend(loc='best', fontsize=11, framealpha=0.9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        safe_metric_name = metric_name.replace("/", "_").replace(" ", "_")
        output_file = os.path.join(output_dir, f"comparison_{algo}_{safe_metric_name}.png")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare algorithms performance across different environments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--log-dir', type=str, default='results/detailed_logs',
                        help='Directory containing JSON log files (default: results/detailed_logs)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for comparison plots')
    parser.add_argument('--env', type=str, default=None,
                        help='Filter by environment name')
    parser.add_argument('--algo', nargs='+', default=None,
                        help='Filter by algorithm names (space-separated)')
    parser.add_argument('--metric', nargs='+', default=None,
                        help='Specific metrics to compare (default: main metrics)')
    parser.add_argument('--group-by', type=str, choices=['env', 'algo', 'all'], default='all',
                        help='Group comparison plots by environment, algorithm, or show all together')
    
    args = parser.parse_args()
    
    print(f"{'='*80}")
    print(f" Loading aggregated experiments from {args.log_dir}")
    print(f"{'='*80}\n")

    aggregated_dir = Path(args.log_dir) / "aggregated_plots"
    experiments = load_aggregated_data(aggregated_dir)
    
    if not experiments:
        print(f"No aggregated experiments found in {aggregated_dir}")
        print(f"Run 'python src/plot_detailed_logs_ALL.py' first to generate aggregated plots")
        return
    
    print(f"Found {len(experiments)} aggregated experiment(s)")

    if args.env:
        experiments = {k: v for k, v in experiments.items() if k[1] == args.env}
        print(f"Filtered to environment: {args.env} ({len(experiments)} experiments)")

    if args.algo:
        experiments = {k: v for k, v in experiments.items() if k[0] in args.algo}
        print(f"Filtered to algorithms: {args.algo} ({len(experiments)} experiments)")
    
    if not experiments:
        print(f"No experiments match the filters")
        return

    print(f"\nLoading metrics data...")
    experiments_data = {}
    for key, exp_info in experiments.items():
        algo, env, t_max, obs_level = key
        print(f"Loading {algo} on {env}...", end=' ')
        
        metrics = load_metrics_from_json_logs(args.log_dir, algo, env, t_max, obs_level)
        if metrics:
            experiments_data[key] = {
                'algo': algo,
                'env': env,
                't_max': t_max,
                'obs_level': obs_level,
                'metrics': metrics
            }
            print(f" ({len(metrics)} metrics)")
        else:
            print(f"No data")
    
    if not experiments_data:
        print(f"\nNo metrics data loaded")
        return

    if args.output_dir is None:
        output_dir = Path(args.log_dir) / "comparison_plots"
    else:
        output_dir = Path(args.output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    if args.metric:
        metrics_to_compare = args.metric
    else:
        metrics_to_compare = [
            "test_return_mean", "return_mean", "test_return_std", "return_std",
            "td_error_abs", "loss", "epsilon", 
            "advantage_mean", "pg_loss", "critic_loss"
        ]

    all_available_metrics = set()
    for data in experiments_data.values():
        all_available_metrics.update(data['metrics'].keys())
    
    metrics_to_compare = [m for m in metrics_to_compare if m in all_available_metrics]
    
    if not metrics_to_compare:
        print(f"\nNo matching metrics found")
        print(f"Available metrics: {sorted(all_available_metrics)}")
        return
    
    print(f"\nGenerating comparison plots for {len(metrics_to_compare)} metric(s)...")
    print(f"Metrics: {', '.join(metrics_to_compare)}")

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    if args.group_by == 'all':
        print(f"\nCreating global comparison plots...")
        for metric_name in metrics_to_compare:
            plot_comparison(experiments_data, metric_name, output_dir, colors)
    
    elif args.group_by == 'env':
        print(f"\nCreating comparison plots grouped by environment...")
        for metric_name in metrics_to_compare:
            plot_comparison_by_env(experiments_data, metric_name, output_dir, colors)
    
    elif args.group_by == 'algo':
        print(f"\nCreating comparison plots grouped by algorithm...")
        for metric_name in metrics_to_compare:
            plot_comparison_by_algo(experiments_data, metric_name, output_dir, colors)
    
    print(f"\n{'='*80}")
    print(f"COMPARISON COMPLETE!")
    print(f"{'='*80}")
    print(f"\nResults location: {output_dir}")
    print()


if __name__ == "__main__":
    main()
