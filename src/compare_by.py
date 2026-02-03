#!/usr/bin/env python3
"""
Génère des plots de comparaison pour les performances par algo, env ou perception.

Usage:
    python src/compare_by.py
    python src/compare_by.py --log-dir results/detailed_logs
    python src/compare_by.py --metric test_return_mean
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import os
from collections import defaultdict
import matplotlib.colors as mcolors


def hex_to_rgb(hex_color):
    """Convertit une couleur hex en RGB (0-1)"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def rgb_to_hex(rgb):
    """Convertit une couleur RGB (0-1) en hex"""
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))


def lighten_color(hex_color, factor=0.3):
    """Éclaircit une couleur en mélangeant avec du blanc"""
    rgb = hex_to_rgb(hex_color)
    lightened = tuple(rgb[i] + (1 - rgb[i]) * factor for i in range(3))
    return rgb_to_hex(lightened)


def darken_color(hex_color, factor=0.3):
    """Assombrit une couleur"""
    rgb = hex_to_rgb(hex_color)
    darkened = tuple(rgb[i] * (1 - factor) for i in range(3))
    return rgb_to_hex(darkened)


def get_perception_base_colors():
    """Retourne les couleurs de base pour chaque perception"""
    return {
        'obs': "#0095ff",           # Bleu
        'state': "#ff7700",         # Orange
        'joint_obs': "#00ff00",     # Vert
        'maro': "#ff0000",          # Rouge
        'md': "#8400ff",            # Violet
        'maro_masks': "#f6c8bf",    # Marron
        'maro_masks_acc': "#ff44c7", # Rose
        'ablation_no_pred': '#7f7f7f', # Gris
        'ablation_no_pred_masks': "#ffff00", # Jaune-vert
        'ablation_no_pred_masks_acc': "#00e5ff", # Cyan
        'masked_joint_obs': "#230f0f", # Cyan
    }


def load_all_metrics(log_dir):
    """Charge toutes les métriques depuis les fichiers JSON """
    log_dir = Path(log_dir)
    json_files = list(log_dir.glob("*_metrics.json"))
    
    all_experiments = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                config = data['config']
                
                experiment = {
                    'algo': config.get('algorithm', 'unknown'),
                    'env': config.get('environment', 'unknown'),
                    'perception': config.get('perception_config', 'unknown'),
                    'seed': config.get('seed', 0),
                    't_max': config.get('t_max', 0),
                    'metrics': data['metrics'],
                    'file': json_file.name
                }
                
                all_experiments.append(experiment)
        except Exception as e:
            print(f"Warning: Failed to load {json_file.name}: {e}")
            continue
    
    print(f"Loaded {len(all_experiments)} experiment(s)")
    return all_experiments


def aggregate_experiments(experiments, group_by_keys):
    """Agrège les expériences par clés """
    grouped = defaultdict(list)
    
    for exp in experiments:
        key = tuple(exp[k] for k in group_by_keys)
        grouped[key].append(exp)
    
    return grouped


def compute_aggregated_metrics(experiments):
    """Calcule les métriques pour un groupe d'expériences"""
    if not experiments:
        return {}
    
    all_metrics = defaultdict(lambda: defaultdict(list))
    
    for exp in experiments:
        for t_str, metrics in exp['metrics'].items():
            if t_str == 'timestamp':
                continue
            
            t = int(t_str)
            
            for metric_name, value in metrics.items():
                if metric_name == 'timestamp':
                    continue
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
            'n_seeds': len(experiments)
        }
    
    return aggregated


def plot_comparison(grouped_data, title, xlabel, ylabel, metric_name, output_file, colors=None):
    """Plot de comparaison"""
    plt.figure(figsize=(14, 8))
    
    has_data = False
    
    for label, metrics in sorted(grouped_data.items()):
        if metric_name not in metrics:
            continue
        
        has_data = True
        metric_data = metrics[metric_name]
        timesteps = metric_data['timesteps']
        mean = np.array(metric_data['mean'])
        std = np.array(metric_data['std'])
        
        # Utiliser la couleur du dictionnaire ou une couleur par défaut
        if colors and isinstance(colors, dict):
            color = colors.get(label, '#000000')
        elif colors:
            color = colors[list(grouped_data.keys()).index(label) % len(colors)]
        else:
            color = plt.cm.tab10(list(grouped_data.keys()).index(label) % 10)

        plt.plot(timesteps, mean, linewidth=2.5, color=color, label=label, alpha=0.9)
        plt.fill_between(timesteps, mean - std, mean + std, alpha=0.2, color=color)
    
    if not has_data:
        plt.close()
        return False
    
    plt.xlabel(xlabel, fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=10, framealpha=0.9, ncol=1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Sauvegarder le plot
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True


def compare_by_algo(experiments, output_dir, metrics_to_plot):
    """Compare les différents environnements et perceptions pour chaque algorithme"""
    print("\n" + "="*80)
    print("COMPARING BY ALGORITHM")
    print("="*80)
    
    algo_dir = Path(output_dir) / "by_algo"

    by_algo = defaultdict(list)
    for exp in experiments:
        by_algo[exp['algo']].append(exp)
    
    perception_base_colors = get_perception_base_colors()
    
    for algo, algo_exps in sorted(by_algo.items()):
        print(f"\nProcessing algorithm: {algo}")
        algo_output_dir = algo_dir / algo
        grouped = aggregate_experiments(algo_exps, ['env', 'perception'])
        
        # Calculer les métriques pour chaque groupe
        aggregated_data = {}
        for (env, perception), group_exps in grouped.items():
            label = f"{env} - {perception}"
            aggregated_data[label] = compute_aggregated_metrics(group_exps)
            print(f"  - {label}: {len(group_exps)} seed(s)")

        for metric_name in metrics_to_plot:
            output_file = algo_output_dir / f"{metric_name.replace('/', '_')}.png"
            title = f"{algo} - {metric_name}"
            
            # Créer un mapping de couleurs pour ce plot avec dégradés par env
            colors_for_plot = {}
            
            # Regrouper par perception pour appliquer des dégradés
            by_perception = defaultdict(list)
            for label in aggregated_data.keys():
                env, perception = label.split(' - ')
                by_perception[perception].append((label, env))
            
            # Attribuer des couleurs avec dégradés
            for perception, labels_envs in by_perception.items():
                base_color = perception_base_colors.get(perception, '#000000')
                n_envs = len(labels_envs)
                
                if n_envs == 1:
                    colors_for_plot[labels_envs[0][0]] = base_color
                else:
                    for i, (label, env) in enumerate(sorted(labels_envs, key=lambda x: x[1])):
                        if i == 0:
                            colors_for_plot[label] = lighten_color(base_color, 0.3)
                        elif i == n_envs - 1:
                            colors_for_plot[label] = darken_color(base_color, 0.3)
                        else:
                            colors_for_plot[label] = base_color
            
            success = plot_comparison(
                aggregated_data,
                title=title,
                xlabel="Timesteps",
                ylabel=metric_name,
                metric_name=metric_name,
                output_file=str(output_file),
                colors=colors_for_plot
            )
            
            if success:
                print(f"     Saved: {output_file}")


def compare_by_env(experiments, output_dir, metrics_to_plot):
    """Compare les différents algorithmes et perceptions pour chaque environnement"""
    print("\n" + "="*80)
    print("COMPARING BY ENVIRONMENT")
    print("="*80)
    
    env_dir = Path(output_dir) / "by_env"

    by_env = defaultdict(list)
    for exp in experiments:
        by_env[exp['env']].append(exp)
    
    perception_base_colors = get_perception_base_colors()
    
    # Couleurs distinctes pour chaque algorithme
    algo_base_colors = {
        'iql_ns': '#1f77b4',      # Bleu
        'mappo_ns': "#ffee00",    # Orange
        'qmix_ns': '#2ca02c',     # Vert
        'ippo_ns': '#d62728',     # Rouge
    }
    
    for env, env_exps in sorted(by_env.items()):
        print(f"\nProcessing environment: {env}")
        env_output_dir = env_dir / env.replace("-", "_")

        grouped = aggregate_experiments(env_exps, ['algo', 'perception'])

        aggregated_data = {}
        for (algo, perception), group_exps in grouped.items():
            label = f"{algo} - {perception}"
            aggregated_data[label] = compute_aggregated_metrics(group_exps)
            print(f"  - {label}: {len(group_exps)} seed(s)")
        
        for metric_name in metrics_to_plot:
            output_file = env_output_dir / f"{metric_name.replace('/', '_')}.png"
            title = f"{env} - {metric_name}"
            
            # Créer un mapping de couleurs pour ce plot
            # Couleur de base = perception, dégradés pour différencier les algos
            colors_for_plot = {}
            
            # Regrouper par perception pour appliquer des dégradés
            by_perception = defaultdict(list)
            for label in aggregated_data.keys():
                algo, perception = label.split(' - ')
                by_perception[perception].append((label, algo))
            
            # Attribuer des couleurs avec dégradés par perception
            for perception, labels_algos in by_perception.items():
                base_color = perception_base_colors.get(perception, '#000000')
                n_algos = len(labels_algos)
                
                if n_algos == 1:
                    colors_for_plot[labels_algos[0][0]] = base_color
                elif n_algos == 2:
                    colors_for_plot[labels_algos[0][0]] = lighten_color(base_color, 0.4)
                    colors_for_plot[labels_algos[1][0]] = darken_color(base_color, 0.3)
                else:
                    # Créer un dégradé distinct pour chaque algo
                    for i, (label, algo) in enumerate(sorted(labels_algos, key=lambda x: x[1])):
                        factor = 0.5 * (i / (n_algos - 1)) if n_algos > 1 else 0
                        if i < n_algos / 2:
                            colors_for_plot[label] = lighten_color(base_color, 0.5 - factor)
                        else:
                            colors_for_plot[label] = darken_color(base_color, factor - 0.25)
            
            success = plot_comparison(
                aggregated_data,
                title=title,
                xlabel="Timesteps",
                ylabel=metric_name,
                metric_name=metric_name,
                output_file=str(output_file),
                colors=colors_for_plot
            )
            
            if success:
                print(f"     Saved: {output_file}")


def compare_by_perception(experiments, output_dir, metrics_to_plot):
    """Compare les différents environnements et algorithmes pour chaque perception"""
    print("\n" + "="*80)
    print("COMPARING BY PERCEPTION")
    print("="*80)
    
    perception_dir = Path(output_dir) / "by_perception"

    by_perception = defaultdict(list)
    for exp in experiments:
        by_perception[exp['perception']].append(exp)
    
    # Définir des couleurs distinctes pour chaque algorithme
    algo_colors = {
        'iql_ns': '#1f77b4',      # Bleu
        'mappo_ns': "#f5e000",    # Orange
        'qmix_ns': '#2ca02c',     # Vert
        'ippo_ns': "#d62727",     # Rouge
    }
    
    for perception, perception_exps in sorted(by_perception.items()):
        print(f"\nProcessing perception: {perception}")
        perception_output_dir = perception_dir / perception

        grouped = aggregate_experiments(perception_exps, ['env', 'algo'])

        aggregated_data = {}
        for (env, algo), group_exps in grouped.items():
            label = f"{env} - {algo}"
            aggregated_data[label] = compute_aggregated_metrics(group_exps)
            print(f"  - {label}: {len(group_exps)} seed(s)")

        for metric_name in metrics_to_plot:
            output_file = perception_output_dir / f"{metric_name.replace('/', '_')}.png"
            title = f"{perception} - {metric_name}"
            
            # Créer un mapping de couleurs pour ce plot
            colors_for_plot = {}
            for label in aggregated_data.keys():
                # Extraire l'algo du label
                algo = label.split(' - ')[-1]
                colors_for_plot[label] = algo_colors.get(algo, '#000000')
            
            success = plot_comparison(
                aggregated_data,
                title=title,
                xlabel="Timesteps",
                ylabel=metric_name,
                metric_name=metric_name,
                output_file=str(output_file),
                colors=colors_for_plot
            )
            
            if success:
                print(f"     Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare experiments by algorithm, environment, or perception',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--log-dir', type=str, default='results/detailed_logs',
                        help='Directory containing JSON log files (default: results/detailed_logs)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for comparison plots (default: log-dir/compare_by)')
    parser.add_argument('--metric', nargs='+', default=None,
                        help='Specific metrics to plot (default: main metrics)')
    parser.add_argument('--group-by', nargs='+', 
                        choices=['algo', 'env', 'perception', 'all'], 
                        default=['all'],
                        help='Which groupings to generate (default: all)')
    
    args = parser.parse_args()
    
    print(f"{'='*80}")
    print(f" COMPARE BY")
    print(f"{'='*80}\n")
    
    # Charger toutes les expériences
    print(f"Loading experiments from: {args.log_dir}")
    experiments = load_all_metrics(args.log_dir)
    
    if not experiments:
        print(f"No experiments found in {args.log_dir}")
        return
    
    # Statistiques
    algos = set(exp['algo'] for exp in experiments)
    envs = set(exp['env'] for exp in experiments)
    perceptions = set(exp['perception'] for exp in experiments)
    
    print(f"\nDataset statistics:")
    print(f"  - Algorithms: {len(algos)} ({', '.join(sorted(algos))})")
    print(f"  - Environments: {len(envs)} ({', '.join(sorted(envs))})")
    print(f"  - Perceptions: {len(perceptions)} ({', '.join(sorted(perceptions))})")
    print(f"  - Total experiments: {len(experiments)}")

    if args.output_dir is None:
        output_dir = Path(args.log_dir) / "compare_by"
    else:
        output_dir = Path(args.output_dir)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Métriques à tracer
    if args.metric:
        metrics_to_plot = args.metric
    else:
        # Métriques par défaut
        metrics_to_plot = [
            # Métriques de performance
            "test_return_mean",
            "test_return_std", 
            "return_mean",
            "return_std",
            # Métriques Q-learning / Value-based
            "td_error_abs",
            "target_mean",
            "q_taken_mean",
            "loss",
            # Métriques Actor-Critic / Policy gradient
            "advantage_mean",
            "pg_loss",
            "critic_loss",
            "pi_max",
            # Métriques de gradient
            "agent_grad_norm",
            "critic_grad_norm",
            "grad_norm",
        ]
    
    # Métriques disponibles
    all_available_metrics = set()
    for exp in experiments:
        for metrics_dict in exp['metrics'].values():
            all_available_metrics.update(metrics_dict.keys())
    all_available_metrics.discard('timestamp')
    
    metrics_to_plot = [m for m in metrics_to_plot if m in all_available_metrics]
    
    if not metrics_to_plot:
        print(f"\nNo matching metrics found")
        print(f"Available metrics: {sorted(all_available_metrics)}")
        return
    
    print(f"\nMetrics to plot: {', '.join(metrics_to_plot)}")
    
    # Par groupe
    group_by = args.group_by
    if 'all' in group_by:
        group_by = ['algo', 'env', 'perception']
    
    if 'algo' in group_by:
        compare_by_algo(experiments, output_dir, metrics_to_plot)
    
    if 'env' in group_by:
        compare_by_env(experiments, output_dir, metrics_to_plot)
    
    if 'perception' in group_by:
        compare_by_perception(experiments, output_dir, metrics_to_plot)
    
    print(f"\n{'='*80}")
    print(f"COMPARISON COMPLETE!")
    print(f"{'='*80}")
    print(f"\nResults location: {output_dir}")
    print(f"  - By algorithm: {output_dir / 'by_algo'}")
    print(f"  - By environment: {output_dir / 'by_env'}")
    print(f"  - By perception: {output_dir / 'by_perception'}")
    print()


if __name__ == "__main__":
    main()
