from dataclasses import dataclass
from typing import NewType
from typing import Optional


@dataclass
class UntrainedModel:
    id: str
    classes: list[str]
    is_grouped: bool
    

ProcessedName = NewType("ProcessedName", list[int, list[int]])


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


