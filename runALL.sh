#!/bin/bash
# Python version = 3.8.10

ENVS=("SimpleSpeakerListener-v0" "SimpleSpeakerListener6-v0" "SimpleSpeakerListener12-v0" "SimpleSpreadXY-v0" "SimpleSpreadXY4-v0" "SimpleSpreadXY8-v0" "SimpleSpreadBlind-v0" "SimpleSpreadBlind6-v0" "SimpleSpreadBlind12-v0")
#ENVS=("SimpleSpeakerListener-v0" "SimpleSpeakerListener6-v0" "SimpleSpeakerListener12-v0")
#ENVS=("SimpleSpreadXY-v0")
ALGOS=("iql_ns" "mappo_ns" "qmix_ns" "ippo_ns")
#ALGOS=("qmix_ns" "ippo_ns")
PERCEPTIONS=("obs" "maro" "state" "joint_obs" "md" "maro_masks" "maro_masks_acc" "ablation_no_pred" "ablation_no_pred_masks" "ablation_no_pred_masks_acc")
#PERCEPTIONS=("maro" "joint_obs" "maro_masks" "ablation_no_pred" "ablation_no_pred_masks")
#PERCEPTIONS=("maro" "maro_masks" "maro_masks_acc")


TIME_LIMIT=25

for ENV in "${ENVS[@]}"
do
   for PERCEPTION in "${PERCEPTIONS[@]}"
   do
      for ALGO in "${ALGOS[@]}"
      do
         echo ""
         echo "=========================================="
         echo "Configuration: ENV=$ENV, ALGO=$ALGO, PERCEPTION=$PERCEPTION"
         echo "=========================================="

         CONFIG_SOURCE="src/config/custom_configs/${PERCEPTION}.yaml"
         
         if [ ! -f "$CONFIG_SOURCE" ]; then
            echo "Warning: Config file $CONFIG_SOURCE not found, skipping..."
            continue
         fi
         
         cp $CONFIG_SOURCE src/config/perception.yaml
         #modifier le print, par défaut il écrit a la fin du perception.yaml
         #écrire après modèle_type serait plus propre
         sed -i.bak '/^config_name:/d' src/config/perception.yaml
         printf "\nconfig_name: \"%s\"\n" "$PERCEPTION" >> src/config/perception.yaml
         rm -f src/config/perception.yaml.bak

         for SEED in {0..2}
         do
            echo "Running seed=$SEED..."
            #python3 src/main.py --config=$ALGO --env-config=gymma with env_args.key=$ENV env_args.time_limit=$TIME_LIMIT t_max=1000000 seed=$SEED
            python3 src/main.py --config=$ALGO --env-config=gymma with env_args.key=$ENV env_args.time_limit=$TIME_LIMIT t_max=10000000 log_interval=10000 runner_log_interval=10000 learner_log_interval=10000 seed=$SEED
            sleep 2s
         done  

         if [ $? -ne 0 ]; then
            echo "Error: Run failed for ENV=$ENV, ALGO=$ALGO, PERCEPTION=$PERCEPTION, seed=$SEED"
         else
            echo "Completed: ENV=$ENV, ALGO=$ALGO, PERCEPTION=$PERCEPTION, seed=$SEED"
         fi
         
         
         echo "Finished configuration: ENV=$ENV, ALGO=$ALGO, PERCEPTION=$PERCEPTION"
         echo ""
      done
   done
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
