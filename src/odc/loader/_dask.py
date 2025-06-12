"""
odc.loader._dask
================

This module provides utility functions for working with Dask, particularly
for tokenizing streams of objects to assist in creating stable Dask graphs.
"""

from typing import (
    Any,
    Callable,
    Hashable,
    Iterator,
    MutableMapping,
    Optional,
    Tuple,
    TypeVar,
)

from dask.base import tokenize

T = TypeVar("T")


def tokenize_stream(
    xx: Iterator[T],
    key: Optional[Callable[[str], Hashable]] = None,
    dsk: Optional[MutableMapping[Hashable, Any]] = None,
) -> Iterator[Tuple[Hashable, T]]:
    """
    Convert a stream of objects `xx` into a stream of `(key, object)` tuples,
    where `key` is derived from `dask.base.tokenize(object)`.

    Optionally populates a Dask graph dictionary `dsk` with these keys and objects.

    :param xx: An iterator of objects to tokenize.
    :param key: An optional function to transform the default string token.
                The input to this function is `tokenize(x)`, and its output
                is used as the key in the yielded tuple and in the `dsk` graph.
    :param dsk: An optional mutable mapping (e.g., a dict) to store the
                tokenized objects. If provided, `dsk[key]` will be set to `x`.
    :yields: Tuples of (hashable_key, original_object).
    """
    if key:
        kx = ((key(tokenize(x)), x) for x in xx)
    else:
        kx = ((tokenize(x), x) for x in xx)

    if dsk is None:
        yield from kx
    else:
        for k, x in kx:
            dsk[k] = x
            yield k, x
