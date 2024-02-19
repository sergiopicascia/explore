"""Utilities functions
"""

from collections.abc import Iterable
from itertools import islice


def batched(iterable: Iterable, batch_size: int):
    """Batch data from iterable into tuples of length batch_size

    Args:
        iterable (Iterable): iterable object to be split
        batch_size (int): size of batch

    Yields:
        batch: tuple of length batch_size
    """
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, batch_size)):
        yield batch
