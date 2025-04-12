from collections.abc import Iterable
from typing import Any
from typing import Callable
from unittest.mock import patch

import pytest

from torchsmith.utils.dtypes import T
from torchsmith.utils.dtypes import X
from torchsmith.utils.dtypes import Y
from torchsmith.utils.pyutils import batch_function
from torchsmith.utils.pyutils import batched
from torchsmith.utils.pyutils import get_arguments


@pytest.mark.parametrize(
    ["x", "n", "expected_output"],
    [([1, 2, 3, 4], 2, [(1, 2), (3, 4)]), ([1, 2, 3, 4], 3, [(1, 2, 3), (4,)])],
)
@pytest.mark.parametrize("python_version", [(3, 12), (3, 11)])
def test_batched(
    x: Iterable[T], n: int, expected_output: list[T], python_version: tuple[int, int]
) -> None:
    with patch("sys.version_info", python_version):
        assert list(batched(x, n)) == expected_output


@pytest.mark.parametrize(
    ["x", "func", "expected_output"],
    [
        ([1, 2, 3, 4], lambda x: [-item for item in x], [[-1, -2], [-3, -4]]),
    ],
)
def test_batch_function(
    func: Callable[[Iterable[X]], Y], x: Iterable[X], expected_output: list[T]
) -> None:
    assert (
        list(batch_function(func=func, input=x, n_jobs=2, batch_size=2))
        == expected_output
    )


def example_func(a: int, b: int = 2, c: int = 3) -> None:
    raise NotImplementedError()


class ExampleClass:
    def __init__(self, x: int, y: int = 10) -> None:
        self.x = x
        self.y = y

    def method(self, a: int, b: int = 5) -> None:
        raise NotImplementedError()

    @classmethod
    def cls_method(cls, value: int = 42) -> None:
        raise NotImplementedError()


@pytest.mark.parametrize(
    ["func", "args", "kwargs", "expected"],
    [
        # Regular function.
        (example_func, (1,), {}, {"a": 1, "b": 2, "c": 3}),
        (example_func, (1, 4), {}, {"a": 1, "b": 4, "c": 3}),
        (example_func, (1,), {"c": 10}, {"a": 1, "b": 2, "c": 10}),
        # Class __init__.
        (ExampleClass.__init__, (None, 5), {}, {"x": 5, "y": 10}),
        (ExampleClass.__init__, (None, 5, 20), {}, {"x": 5, "y": 20}),
        # Instance method.
        (ExampleClass.method, (None, 3), {}, {"a": 3, "b": 5}),
        (ExampleClass.method, (None, 3, 7), {}, {"a": 3, "b": 7}),
        # Class method.
        (ExampleClass.cls_method, (), {}, {"value": 42}),
        (ExampleClass.cls_method, (), {"value": 99}, {"value": 99}),
    ],
)
def test_get_arguments(
    func: Callable, args: tuple[Any], kwargs: dict[str, Any], expected: dict[str, Any]
) -> None:
    assert get_arguments(func, *args, **kwargs) == expected
