from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from logger import logger

class Dataset(Enum):
    TRAIN = "train"
    VALIDATION = "validation"


@dataclass
class Scores:
    f1: float
    precision: float
    recall: float


@dataclass
class Metrics:
    accuracy: float
    loss: Optional[float] = None
    scores: Optional[Scores] = None


@dataclass
class TrainHistory:
    train: list[Metrics] = field(default_factory=list)
    validation: list[Metrics] = field(default_factory=list)


@dataclass
class TrainLogger:
    train_history: TrainHistory = field(default_factory=TrainHistory)
    results: Optional[Metrics] = None

    def log_epoch(self, epoch: int):
        epoch_train_metrics = self.train_history.train[epoch - 1]
        epoch_val_metrics = self.train_history.validation[epoch - 1]

        logger.info(f"Epoch: {epoch}, Train Acc: {epoch_train_metrics.accuracy}, "
              f"Train Loss: {epoch_train_metrics.loss}, Val Acc: {epoch_val_metrics.accuracy}, "
              f"Val Loss: {epoch_val_metrics.loss}")

    def save_epoch(self, metrics: Metrics, dataset: Dataset):
        if dataset == Dataset.TRAIN:
            self.train_history.train.append(metrics)
        else:
            self.train_history.validation.append(metrics)

    def save_test_evaluation(self, metrics: Metrics):
        metrics.loss = None
        self.results = metrics

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)
