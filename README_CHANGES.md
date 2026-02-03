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
