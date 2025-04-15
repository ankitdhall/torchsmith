import inspect
import os
import resource
import sys
from collections.abc import Iterable
from collections.abc import Iterator
from itertools import islice
from typing import Any
from typing import Callable

import psutil
from joblib import Parallel
from joblib import delayed

from torchsmith.utils.dtypes import T
from torchsmith.utils.dtypes import X
from torchsmith.utils.dtypes import Y


def batched(iterable: Iterable[T], n: int) -> Iterator[tuple[T, ...]]:
    print(f"\n\n\nsys.version_info: {sys.version_info}")
    if sys.version_info >= (3, 12):
        from itertools import batched as itertools_batched

        yield from itertools_batched(iterable, n)
    else:
        if n < 1:
            raise ValueError("n must be at least one")
        it = iter(iterable)
        while batch := tuple(islice(it, n)):
            yield batch


def batch_function(
    *,
    func: Callable[[Iterable[X]], Y],
    input: Iterable[X],
    n_jobs: int,
    batch_size: int,
) -> Iterator[Y]:
    batches = batched(input, batch_size)
    return (
        batch_result
        for bs in iter(
            lambda b=batches: list(islice(b, n_jobs)),  # type: ignore
            [],
        )
        # Keeps pulling batches until empty
        if bs  # Ensures we don't pass empty batches
        for batch_result in Parallel(n_jobs=len(bs), return_as="generator")(
            delayed(lambda x: func(x))(list(batch)) for batch in bs
        )
    )


def get_arguments(
    func: Callable, *args: Any, **kwargs: dict[str, Any]
) -> dict[str, Any]:
    fn_signature = inspect.signature(func)
    bound_args = fn_signature.bind(*args, **kwargs)
    bound_args.apply_defaults()

    # If it's a method (class or instance), exclude 'self' or 'cls'
    return {k: v for k, v in bound_args.arguments.items() if k not in {"self", "cls"}}


def set_resource_limits(n_jobs: int | None, *, maximum_memory: int | None) -> None:
    if maximum_memory is not None:
        # Set the maximum RAM + swap size to 1 GB (1024 * 1024 * 1024 bytes)
        resource.setrlimit(resource.RLIMIT_AS, (26 * 1024 * 1024 * 1024, -1))
        print(f"Maximum AS to {maximum_memory} GB")

    if n_jobs is not None:
        # Limit which cores are used.
        pid = os.getpid()
        psutil.Process(pid).cpu_affinity(list(range(n_jobs)))
        print(
            f"Process {pid} is now limited to CPU cores: "
            f"{psutil.Process(pid).cpu_affinity()}"
        )
