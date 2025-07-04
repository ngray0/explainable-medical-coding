import os
import re
from pathlib import Path
from typing import Any

import torch

# load environment variables
from dotenv import find_dotenv, load_dotenv
from omegaconf import OmegaConf

import wandb

load_dotenv(find_dotenv())

experiment_path = os.environ["EXPERIMENT_PATH"]

FORMATTING_PATTERN = r"\[([^\]]+)\]"
FORMATTING_REGEX = re.compile(FORMATTING_PATTERN)


def length_of_formatting(string: str):
    return sum(
        len(s) + 2 for s in FORMATTING_REGEX.findall(string)
    )  # plus 2 for parenthesis


def length_without_formatting(string: str):
    return len(string) - length_of_formatting(string)


def source_string(source):
    return f"{source[:18]}.." if len(source) > 20 else f"{source}"


class BaseCallback:
    def __init__(self):
        pass

    def on_initialisation_end(self, trainer=None):
        pass

    def on_train_begin(self, trainer=None):
        pass

    def on_train_end(self, trainer=None):
        pass

    def on_val_begin(self, trainer=None):
        pass

    def on_val_end(self, trainer=None):
        pass

    def on_epoch_begin(self, trainer=None):
        pass

    def on_epoch_end(self, trainer=None):
        pass

    def on_batch_begin(self, trainer=None):
        pass

    def on_batch_end(self, trainer=None):
        pass

    def on_fit_begin(self, trainer=None):
        pass

    def on_fit_end(self, trainer=None):
        pass

    def log_dict(
        self, nested_dict: dict[str, dict[str, torch.Tensor]], epoch: int
    ) -> None:
        pass

    def on_end(self, trainer=None):
        pass


class WandbCallback(BaseCallback):
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.config = config

    def extract_tags(self, trainer) -> list[str]:
        tags = []
        tags.append(trainer.config.model.name)
        tags.append(Path(trainer.config.data.dataset_path).stem)
        return tags

    def on_initialisation_end(self, trainer=None):
        wandb_cfg = OmegaConf.to_container(
            trainer.config, resolve=True, throw_on_missing=True
        )
        tags = self.extract_tags(trainer)
        if trainer.config.debug:
            mode = "disabled"
        else:
            mode = "online"

        wandb_cfg["num_parameters"] = sum(p.numel() for p in trainer.model.parameters())
        wandb_cfg["num_trainable_parameters"] = sum(
            p.numel() for p in trainer.model.parameters() if p.requires_grad
        )

        if trainer.lookups is not None:
            wandb_cfg["data_info"] = trainer.lookups.data_info
            OmegaConf.save(trainer.config, trainer.experiment_path / "config.yaml")

        wandb.init(
            config=wandb_cfg,
            settings=wandb.Settings(start_method="thread"),
            tags=tags,
            name=trainer.config.name,
            mode=mode,
            **self.config,
        )
        wandb.watch(trainer.model)

        if not trainer.config.debug:
            # make a folder for the model where configs and model weights are saved

            trainer.experiment_path = Path(experiment_path) / wandb.run.id
            trainer.experiment_path.mkdir(exist_ok=False, parents=True)

    def log_dict(
        self,
        nested_dict: dict[str, Any],
        epoch: int,
    ) -> None:
        # Flatten nested results for better wandb visualization
        flattened_dict = self._flatten_results_for_wandb(nested_dict)
        flattened_dict["epoch"] = epoch
        wandb.log(flattened_dict)  # type: ignore  # noqa

    def _flatten_results_for_wandb(self, nested_dict: dict[str, Any]) -> dict[str, Any]:
        """Flatten nested results dictionary for better wandb logging."""
        flattened = {}
        
        for split_name, split_results in nested_dict.items():
            if split_name == "epoch":
                continue
                
            if isinstance(split_results, dict):
                # Check if this is the new nested structure (with code systems)
                if any(isinstance(v, dict) for v in split_results.values()):
                    # New structure: split -> code_system -> metrics
                    for code_system, metrics in split_results.items():
                        if isinstance(metrics, dict):
                            for metric_name, metric_value in metrics.items():
                                key = f"{split_name}/{code_system}/{metric_name}"
                                flattened[key] = metric_value
                        else:
                            key = f"{split_name}/{code_system}"
                            flattened[key] = metrics
                else:
                    # Old structure: split -> metrics
                    for metric_name, metric_value in split_results.items():
                        key = f"{split_name}/{metric_name}"
                        flattened[key] = metric_value
            else:
                # Direct value
                flattened[split_name] = split_results
                
        return flattened

    def on_end(self, trainer=None):
        wandb.finish()


class SaveBestModelCallback(BaseCallback):
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.config = config
        self.prev_best = None
        self.split_name = config.split
        self.target_name = config.target
        self.metric_name = config.metric

    def on_epoch_end(self, trainer=None):
        best_metric = trainer.metric_collections[self.split_name][self.target_name].get_best_metric(
            self.metric_name
        )
        if self.prev_best is None or best_metric != self.prev_best:
            self.prev_best = best_metric
            trainer.save_checkpoint("best_model.pt")
            print("Saved best model")

    def on_fit_end(self, trainer=None):
        trainer.load_checkpoint("best_model.pt")
        print("Loaded best model")


class EarlyStoppingCallback(BaseCallback):
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.config = config
        self.split_name = config.split
        self.target_name = config.target
        self.metric_name = config.metric
        self.patience = config.patience
        self.counter = 0
        self.prev_best = None

    def on_epoch_end(self, trainer=None):
        """On the end of each epoch, test if the validation metric has improved. If it hasn't improved for self.patience epochs, stop training.

        Args:
            trainer (Trainer, optional): Trainer class. Defaults to None.
        """
        best_metric = trainer.metric_collections[self.split_name][self.target_name].get_best_metric(
            self.metric_name
        )
        if self.prev_best is None or best_metric != self.prev_best:
            self.prev_best = best_metric
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            trainer.stop_training = True
            print(
                f"Early stopping: {self.counter} epochs without improvement for {self.metric_name}"
            )
