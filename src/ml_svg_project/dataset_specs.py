from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DatasetSpec:
    name: str
    max_rows: int | None = None
    streaming: bool = False

    def identity(self) -> str:
        identity = self.name
        if self.max_rows is not None:
            identity += f"__maxrows_{self.max_rows}"
        if self.streaming:
            identity += "__stream"
        return identity


def parse_dataset_spec(value: Any) -> DatasetSpec:
    if isinstance(value, str):
        return DatasetSpec(name=value)
    if isinstance(value, dict):
        name = value.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Dataset spec dict must include a non-empty 'name'.")
        max_rows = value.get("max_rows")
        if max_rows is not None:
            max_rows = int(max_rows)
            if max_rows <= 0:
                raise ValueError("Dataset spec 'max_rows' must be positive when provided.")
        streaming = bool(value.get("streaming", False))
        return DatasetSpec(name=name, max_rows=max_rows, streaming=streaming)
    raise ValueError("Dataset spec must be either a string or a dict.")


def parse_dataset_specs(dataset_cfg: dict[str, Any]) -> list[DatasetSpec]:
    raw_specs = [dataset_cfg["primary"], *dataset_cfg.get("supplementary", [])]
    return [parse_dataset_spec(spec) for spec in raw_specs]


def dataset_names_from_specs(specs: list[DatasetSpec]) -> list[str]:
    return [spec.name for spec in specs]


def dataset_identities_from_specs(specs: list[DatasetSpec]) -> list[str]:
    return [spec.identity() for spec in specs]
