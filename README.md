# Centralized training with hybrid execution in multi-agent reinforcement learning

> **CTDE with a hybrid MARO (Multi-Agent Reinforcement with Observation reconstruction) approach for Multi-Agent Reinforcement Learning (MARL).**

## Installation instructions

To install the codebase, run (tested with python 3.8.10): 
```sh
./install.sh
```

## Running experiments
After installation, you can use the script run.sh to run experiments, where:
1) `ENV` variable selects the environment to use:
    - `SimpleSpreadXY-v0` - SpreadXY-2
    - `SimpleSpreadXY4-v0` - SpreadXY-4;
    - `SimpleSpreadBlind-v0` - SpreadBlindfold;
    - `SimpleBlindDeaf-v0` - HearSee;
    - `SimpleSpread-v0` - SimpleSpread;
    - `SimpleSpeakerListener-v0` - SimpleSpeakerListener;
    - `Foraging-2s-15x15-2p-2f-coop-v2` - LBF environment (modified);
2) `ALGO` variable selects the RL algorithm to use (`iql_ns`, `qmix_ns`, `ippo_ns`, or `mappo_ns` for MPE environments; `iql_ns_lbf`, `qmix_ns_lbf`, `ippo_ns_lbf`, or `mappo_ns_lbf` for LBF environments).
3)  `PERCEPTION` variable selects the perceptual model to use:
    - `obs` - Obs.
    - `joint_obs` - Oracle;
    - `joint_obs_drop_test` - Masked joint obs.;
    - `ablation_no_pred` - MD baseline;
    - `ablation_no_pred_masks` - MD w/ masks baseline;
    - `maro_no_training` - MARO;
    - `maro` - MARO w/ dropout;
4) `TIME_LIMIT` variable: 25 for MPE environments; 30 for LBF environments.



# Modifications du projet Hybrid-MARL

## Ce projet est un FORK réalisé par :
- Tom Bouscarat (@Chef-du-Jardin)

## Modifications principales

### 1. Système de logging JSON (`src/run.py` et `src/utils/logging.py`)

- **Ajout du logging détaillé** : Les métriques d entraînement sont maintenant sauvegardées dans des fichiers JSON structurés
- **Fichiers générés** :
  - `*_detailed.log` : Logs textuels détaillés avec timestamps
  - `*_metrics.json` : Métriques d entraînement au format JSON pour analyse
- **Nomenclature** : Les fichiers incluent l algo, le niveau d obs et l env dans leur nom
  - Format : `{algo}_{perception}_seed{N}_{env}_{timestamp}_metrics.json`
  - Exemple : `iql_ns_joint_obs_seed0_SimpleSpeakerListener-v0_2026-01-28 05:11:50.681664_metrics.json`

### 2. Scripts d'automatisation

#### `runALL.sh`
Script pour exécuter automatiquement toutes les combinaisons d expériences :
- Environnements multiples
- Algorithmes multiples (IPPO, IQL, MAPPO, QMIX, etc.)
- Perceptions multiples (obs, maro, joint_obs, masked_joint_obs, etc.)
- 3 seeds par configuration

### 3. Scripts de visualisation et comparaison

#### `src/generate_plots.py`
Génère des plots individuels pour chaque expérience :
- Plots par seed
- Plots agrégés (moyenne ± écart-type sur toutes les seeds)
- Sauvegarde dans `results/detailed_logs/seed_plots/` et `aggregated_plots/`

#### `src/compare_algorithms.py`
Compare les performances de différents algorithmes :
- Comparaison globale

#### `src/compare_by.py`
Comparaisons multi-dimensionnelles organisées par :
- **by_algo/** : Compare différents environnements et perceptions pour chaque algorithme
- **by_env/** : Compare différents algorithmes et perceptions pour chaque environnement
- **by_perception/** : Compare différents environnements et algorithmes pour chaque perception

Génère automatiquement des sous-dossiers organisés avec tous les plots de comparaison.

## Utilisation

```bash
# Lancer toutes les expériences
./runALL.sh

# Générer les plots individuels et agrégés
python src/generate_plots.py

# Générer les comparaisons multi-dimensionnelles
python src/compare_by.py

# Comparaisons spécifiques
python src/compare_by.py --group-by algo
python src/compare_by.py --group-by env perception
python src/compare_by.py --metric test_return_mean return_mean
```

## Structure des résultats

```
results/detailed_logs/
├── *_metrics.json                 # Métriques JSON par expérience
├── *_detailed.log                 # Logs détaillés par expérience
├── seed_plots/                    # Plots individuels par seed
├── aggregated_plots/              # Plots agrégés par configuration
├── comparison_plots/              # Comparaisons globales
└── compare_by/                    # Comparaisons multi-dimensionnelles
    ├── by_algo/                   # Par algorithme
    ├── by_env/                    # Par environnement
    └── by_perception/             # Par perception
```
