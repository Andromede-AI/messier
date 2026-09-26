from collections.abc import Iterator
from typing import TYPE_CHECKING
import pandas as pd
from .io import data_path, load_jsonl
from .models import (
    DatasetBuilderConfig,
    IRTConfig,
    IRTModelType,
    Record,
    RecordType,
    Task,
    Trajectory,
    VerifierDefinition,
)

if TYPE_CHECKING:
    from .build import DatasetBuilder
    from .irt import IRT


def __getattr__(name: str):

    if name == "DatasetBuilder":
        try:
            from .build import DatasetBuilder
        except ModuleNotFoundError as error:
            raise ImportError("DatasetBuilder requires the development tools. Run `uv sync --group dev`.") from error

        return DatasetBuilder

    if name == "IRT":
        try:
            from .irt import IRT
        except ModuleNotFoundError as error:
            raise ImportError("IRT requires the development tools. Run `uv sync --group dev`.") from error

        return IRT

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def records(source: str = "hf") -> pd.DataFrame:
    """
    Load stored trial results, verifier results, and source summaries.

    :param source: ``hf`` for the Hugging Face release or ``local`` for the current build.
    :return: One row per MESSIER record.
    """
    return pd.read_json(data_path("records.jsonl", source), lines=True)


def tasks(source: str = "hf") -> pd.DataFrame:
    """
    Load task definitions.

    :param source: ``hf`` for the Hugging Face release or ``local`` for the current build.
    :return: One row per task.
    """
    return pd.read_json(data_path("tasks.jsonl", source), lines=True)


def verifiers(source: str = "hf") -> pd.DataFrame:
    """
    Load verifier definitions.

    :param source: ``hf`` for the Hugging Face release or ``local`` for the current build.
    :return: One row per verifier.
    """
    return pd.read_json(data_path("verifiers.jsonl", source), lines=True)


def trajectories(source: str = "hf") -> Iterator[Trajectory]:
    """
    Load source-provided agent trajectories.

    :param source: ``hf`` for the Hugging Face release or ``local`` for the current build.
    :return: Available trajectories, read one at a time.
    """
    for row in load_jsonl(data_path("trajectories.jsonl", source)):
        yield Trajectory.model_validate(row)


def trial_records(records: pd.DataFrame) -> pd.DataFrame:
    """
    Keep the final result for each trial.

    :param records: Data frame loaded from ``records.jsonl``.
    :return: A copy containing those results.
    """
    return records.loc[records["record_type"] == "trial_result"].reset_index(drop=True)


__all__ = [
    "DatasetBuilder",
    "DatasetBuilderConfig",
    "IRT",
    "IRTConfig",
    "IRTModelType",
    "Record",
    "RecordType",
    "Task",
    "Trajectory",
    "VerifierDefinition",
    "records",
    "tasks",
    "trajectories",
    "trial_records",
    "verifiers",
]
