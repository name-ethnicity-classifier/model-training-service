from dataclasses import dataclass
from enum import Enum
import json


@dataclass
class BaseMetrics:
    accuracy: float
    f1: float
    precision: float
    recall: float


@dataclass
class EpochMetrics(BaseMetrics):
    loss: float


class Dataset(Enum):
    TRAIN = "train"
    VALIDATION = "validation"


class LogManager:
    def __init__(self, model_id: str, base_model: str, classes: list[str]):
        self.logs = {
            "model-id": model_id,
            "base-model": base_model,
            "classes": classes,
            "train-history": {
                "train": [],
                "validation": []
            },
            "results": None
        }

    def _create_single_log(self, metrics: BaseMetrics | EpochMetrics):
        return {
            "accuracy": metrics.accuracy,
            "scores": {
                "f1": metrics.f1,
                "precision": metrics.precision,
                "recall": metrics.recall
            },
            "loss": metrics.loss if isinstance(metrics, EpochMetrics) else None
        }

    def log_epoch(self, epoch: int):
        epoch_train_metrics = self.logs["train-history"]["train"][epoch]
        epoch_val_metrics = self.logs["train-history"]["validation"][epoch]

        train_acc = epoch_train_metrics["accuracy"]
        train_loss = epoch_train_metrics["loss"]
        val_acc = epoch_val_metrics["accuracy"]
        val_loss = epoch_val_metrics["loss"]

        print(f"epoch: {epoch}, train-acc: {train_acc}, train-loss: {train_loss}, val-acc: {val_acc}, val-loss: {val_loss}")

    def save_epoch(self, metrics: EpochMetrics, dataset: Dataset):
        epoch_log = self._create_single_log(metrics)
        self.logs["train-history"][dataset.value].append(epoch_log)

    def save_test_evaluation(self, metrics: BaseMetrics):
        test_results = self._create_single_log(metrics)
        del test_results["loss"]

        self.logs["results"] = test_results

    def json_serialize(self):
        return json.dumps(self.logs)
