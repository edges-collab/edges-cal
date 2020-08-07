"""Tools to use in other modules."""
from itertools import product
from typing import List


def dct_of_list_to_list_of_dct(dct: dict) -> List:
    """Take a dict of key: list pairs and turn it into a list of all combinations of dicts.

    Parameters
    ----------
    dct : dict
        A dictionary for which each value is an iterable.

    Returns
    -------
    list :
        A list of dictionaries, each having the same keys as the input ``dct``, but
        in which the values are the elements of the original iterables.

    Examples
    --------
    >>> dct_of_list_to_list_of_dct(
    >>>    { 'a': [1, 2], 'b': [3, 4]}
    [
        {"a": 1, "b": 3},
        {"a": 1, "b": 4},
        {"a": 2, "b": 3},
        {"a": 2, "b": 4},
    ]
    """
    lists = dct.values()

    prod = product(*lists)

    return [{k: v for k, v in zip(dct.keys(), p)} for p in prod]
