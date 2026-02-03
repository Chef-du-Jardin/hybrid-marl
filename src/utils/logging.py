from collections import defaultdict
import logging
import numpy as np
import os
from datetime import datetime
import json

class Logger:
    def __init__(self, console_logger):
        self.console_logger = console_logger

        self.use_tb = False
        self.use_sacred = False
        self.use_hdf = False

        self.stats = defaultdict(lambda: [])

        self.detailed_log_file = None
        self.log_file_path = None

        self.json_log_file = None
        self.json_log_path = None
        self.json_data = {"config": {}, "metrics": {}}

    def setup_tb(self, directory_name):
        # Import here so it doesn't have to be installed if you don't use it
        from tensorboard_logger import configure, log_value
        configure(directory_name)
        self.tb_logger = log_value
        self.use_tb = True

    def setup_sacred(self, sacred_run_dict):
        self._run_obj = sacred_run_dict
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = True

    def setup_detailed_log(self, log_dir, experiment_name):
        """Configure le fichier de log """
        os.makedirs(log_dir, exist_ok=True)
        self.log_file_path = os.path.join(log_dir, f"{experiment_name}_detailed.log")
        self.detailed_log_file = open(self.log_file_path, 'w', buffering=1)
        self.detailed_log_file.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.console_logger.info(f"Detailed log file: {self.log_file_path}")

    def setup_json_log(self, log_dir, experiment_name, config):
        """Configure les métriques"""
        os.makedirs(log_dir, exist_ok=True)
        self.json_log_path = os.path.join(log_dir, f"{experiment_name}_metrics.json")
        perception_args = config.get("perception_args", {})
        self.json_data["config"] = {
            "experiment_name": experiment_name,
            "algorithm": config.get("name", "unknown"),
            "environment": config.get("env_args", {}).get("key", "unknown"),
            "seed": config.get("seed", 0),
            "observation_level": perception_args.get("model_type", "full_obs"),
            "perception_config": perception_args.get("config_name", perception_args.get("model_type", "unknown")),
            "t_max": config.get("t_max", 0),
            "test_interval": config.get("test_interval", 0),
            "log_interval": config.get("log_interval", 0),
            "learning_rate": config.get("lr", 0),
            "batch_size": config.get("batch_size", 0),
            "started_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.json_data["metrics"] = {}
        
        self.console_logger.info(f"JSON metrics file: {self.json_log_path}")

    def log_to_file(self, message, t_env=None):
        """Écrit un log détaillé"""
        if self.detailed_log_file:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if t_env is not None:
                self.detailed_log_file.write(f"[{timestamp}] [t={t_env}] {message}\n")
            else:
                self.detailed_log_file.write(f"[{timestamp}] {message}\n")

    def log_stats_to_file(self, t_env, prefix=""):
        """Écrit toutes les stats dans le fichier de log"""
        if not self.detailed_log_file or not self.stats:
            return
        self.log_to_file(f"{prefix}Stats at t_env={t_env}", t_env)
        
        for (k, v) in sorted(self.stats.items()):
            if len(v) > 0:
                window = 5 if k != "epsilon" else 1
                recent_values = [x[1] for x in v[-window:]]
                try:
                    mean_val = np.mean(recent_values)
                    self.log_to_file(f"  {k}: {mean_val:.4f}", t_env)
                except:
                    try:
                        mean_val = np.mean([x.item() for x in recent_values])
                        self.log_to_file(f"  {k}: {mean_val:.4f}", t_env)
                    except:
                        pass
        
        self.log_to_file("", t_env)

    def log_stats_to_json(self, t_env):
        """Sauvegardeles stats dans le fichier JSON"""
        if self.json_log_path is None or not self.stats:
            return
        t_key = str(t_env)
        self.json_data["metrics"][t_key] = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        for (k, v) in self.stats.items():
            if len(v) > 0:
                window = 5 if k != "epsilon" else 1
                recent_values = [x[1] for x in v[-window:]]
                try:
                    mean_val = float(np.mean(recent_values))
                    self.json_data["metrics"][t_key][k] = mean_val
                except:
                    try:
                        mean_val = float(np.mean([x.item() for x in recent_values]))
                        self.json_data["metrics"][t_key][k] = mean_val
                    except:
                        pass
        try:
            with open(self.json_log_path, 'w') as f:
                json.dump(self.json_data, f, indent=2)
        except Exception as e:
            self.console_logger.warning(f"Failed to write JSON log: {e}")

    def close_detailed_log(self):
        """Verification fermeture JSON"""
        if self.detailed_log_file:
            self.detailed_log_file.write(f"\n{'='*50}\n")
            self.detailed_log_file.write(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.detailed_log_file.close()
            self.detailed_log_file = None
        if self.json_log_path:
            self.json_data["config"]["finished_at"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            try:
                with open(self.json_log_path, 'w') as f:
                    json.dump(self.json_data, f, indent=2)
                self.console_logger.info(f"JSON metrics saved to: {self.json_log_path}")
            except Exception as e:
                self.console_logger.warning(f"Failed to finalize JSON log: {e}")

    def log_stat(self, key, value, t, to_sacred=True):
        self.stats[key].append((t, value))

        if self.use_tb:
            self.tb_logger(key, value, t)

        if self.use_sacred and to_sacred:
            if key in self.sacred_info:
                self.sacred_info["{}_T".format(key)].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info["{}_T".format(key)] = [t]
                self.sacred_info[key] = [value]
            
            self._run_obj.log_scalar(key, value, t)

    def log_model(self, name, filepath, to_sacred=True):

        if self.use_sacred and to_sacred:
            self._run_obj.add_artifact(filepath, filepath)

    def print_recent_stats(self):
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(*self.stats["episode"][-1])
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            try:
                item = "{:.4f}".format(np.mean([x[1] for x in self.stats[k][-window:]]))
            except:
                item = "{:.4f}".format(np.mean([x[1].item() for x in self.stats[k][-window:]]))
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.console_logger.info(log_str)

        if "episode" in self.stats and len(self.stats["episode"]) > 0:
            t_env = self.stats["episode"][-1][0]
            self.log_stats_to_file(t_env)
            self.log_stats_to_json(t_env)


# set up a custom logger
def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('DEBUG')

    return logger

