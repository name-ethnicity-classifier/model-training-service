from dataclasses import dataclass
from typing import NewType


@dataclass
class UntrainedModel:
    id: str
    classes: list[str]
    is_grouped: bool


ProcessedName = NewType("ProcessedName", list[int, list[int]])


