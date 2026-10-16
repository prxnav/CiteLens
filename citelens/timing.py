"""A decorator that times each lazily-evaluated parsing stage."""

from __future__ import annotations

import functools
import time
from collections.abc import Callable, Iterator
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")


def timed(stage: str) -> Callable[[Callable[P, Iterator[T]]], Callable[P, Iterator[T]]]:
    """Log how long a generator-based stage takes to fully drain.

    Wrapping a generator function the naive way only times how long it takes
    to *create* the generator, not to run its body — that body doesn't
    execute until something iterates it. So this decorator returns a new
    generator that forwards each item from the wrapped one and only stops
    the clock once the caller has actually consumed it all (or stopped
    early), via the ``finally`` block.
    """

    def decorator(func: Callable[P, Iterator[T]]) -> Callable[P, Iterator[T]]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> Iterator[T]:
            start = time.perf_counter()
            count = 0
            try:
                for item in func(*args, **kwargs):
                    count += 1
                    yield item
            finally:
                elapsed = time.perf_counter() - start
                print(f"[{stage}] yielded {count} item(s) in {elapsed:.4f}s")

        return wrapper

    return decorator
