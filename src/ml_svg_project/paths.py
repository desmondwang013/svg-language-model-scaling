from pathlib import Path

from .config import ARTIFACTS_DIR, DATA_DIR, OUTPUTS_DIR


def ensure_project_dirs() -> None:
    for path in (
        DATA_DIR,
        ARTIFACTS_DIR,
        OUTPUTS_DIR,
        OUTPUTS_DIR / "figures",
        OUTPUTS_DIR / "samples",
        OUTPUTS_DIR / "reports",
    ):
        path.mkdir(parents=True, exist_ok=True)
