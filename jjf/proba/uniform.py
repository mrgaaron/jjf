from collections.abc import Coroutine
import numpy as np
from typing import TypeVar

T = TypeVar("T")


def draw(
    space: list[T], size_min: int = 1, size_max: int = 1, n: int = 1
) -> Coroutine[list[T], int, int, int, list[list[T]]]:
    """Draw n samples from a uniform distribution without replacement.
    size_min and size_max indicate the minimum and maximum dimensions of
    each sample.  For example, if size_min is 1 and size_max is 3, then over
    n samples, 1-3 elements will be drawn from the space.

    Args:
        space (list): the space to sample from.
        size_min (int): the minimum elements in each sample.
        size_max (int): the maximum elements in each sample.
        n (int): the number of samples to draw.

    Returns:
        list: The samples drawn from the uniform distribution.

    """

    for _ in range(n):
        size = np.random.randint(size_min, size_max + 1)
        yield np.random.choice(space, size=size, replace=False).tolist()
