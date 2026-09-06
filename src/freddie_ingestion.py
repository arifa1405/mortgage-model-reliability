"""Functions for loading Freddie Mac annual sample files."""

from pathlib import Path

import pandas as pd

from .freddie_config import (
    ORIGINATION_COLUMNS,
    ORIGINATION_STRING_COLUMNS,
)


def load_origination(
    file_path: str | Path,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Load a Freddie Mac pipe-delimited origination file."""

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Origination file not found: {file_path}"
        )

    origination = pd.read_csv(
        file_path,
        sep="|",
        header=None,
        names=ORIGINATION_COLUMNS,
        dtype=ORIGINATION_STRING_COLUMNS,
        nrows=nrows,
        low_memory=False,
    )

    if origination.shape[1] != len(ORIGINATION_COLUMNS):
        raise ValueError(
            "Unexpected number of origination columns: "
            f"{origination.shape[1]}"
        )

    return origination