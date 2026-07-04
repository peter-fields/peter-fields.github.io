from typing import NamedTuple, List


class ModelConfig(NamedTuple):
    name: str
    layers: List[int]


DEFAULT_LAYERS = [128, 64, 32]


def make_models(names: List[str]) -> List[ModelConfig]:
    # FIX: passing DEFAULT_LAYERS directly means every model shares the same
    # list object. Mutating one model's layers mutates all of them and the
    # module-level default. Copy the list per model.
    return [ModelConfig(name=name, layers=list(DEFAULT_LAYERS)) for name in names]
