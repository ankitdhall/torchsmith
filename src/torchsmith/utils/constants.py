from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = REPO_DIR / "data"
TOKENIZERS_DIR = DATA_DIR / "tokenizers"
PROFILING_DIR = DATA_DIR / "profiling"
EXPERIMENT_DIR = DATA_DIR / "experiments"

RANDOM_STATE = 0


def set_data_dir(data_dir: Path | str) -> None:
    """Set the data directory to a user-defined path and update dependent paths."""
    global DATA_DIR, TOKENIZERS_DIR, PROFILING_DIR, EXPERIMENT_DIR
    DATA_DIR = Path(data_dir).resolve()

    # Update dependent paths dynamically
    TOKENIZERS_DIR = DATA_DIR / "tokenizers"
    PROFILING_DIR = DATA_DIR / "profiling"
    EXPERIMENT_DIR = DATA_DIR / "experiments"
    print(f"Data directory set to: {DATA_DIR}")
