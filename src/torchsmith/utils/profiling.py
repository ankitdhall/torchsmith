from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from pyinstrument import Profiler

from torchsmith.utils.constants import PROFILING_DIR


@contextmanager
def profiler(name: str, *, path: str | Path = PROFILING_DIR):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with Profiler() as p:
        yield

    filename = (
        f"profile_report_{name}_{datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')}.html"
    )
    with open(
        path / filename,
        "w",
    ) as f:
        f.write(p.output_html())
