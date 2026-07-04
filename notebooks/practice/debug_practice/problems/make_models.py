from typing import NamedTuple, List


class ModelConfig(NamedTuple):
    name: str
    layers: List[int]


DEFAULT_LAYERS = [128, 64, 32]


def make_models(names: List[str]) -> List[ModelConfig]:
    """Create a ModelConfig for each name, each starting from DEFAULT_LAYERS.

    Each model's `layers` list must be independent: appending to one
    model's layers must not affect any other model, and must not affect
    DEFAULT_LAYERS.
    """
    return [ModelConfig(name=name, layers=DEFAULT_LAYERS.copy()) for name in names]
