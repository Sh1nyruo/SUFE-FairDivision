"""
Classes for representing a Bundle (a set of objects given to an agent).

Designed for indivisible item allocation with additive valuation functions.

Author: Gu Yongwei
Since: 2024-02
"""
from abc import ABC
from typing import List, Any, Dict
from collections.abc import Iterable

DEFAULT_SEPARATOR = ","   # separator between items in printing

class Bundle(ABC):
    def __repr__(self):
        separator = DEFAULT_SEPARATOR
        if self.items is None:
            return "None"
        else:
            return "{" + separator.join(map(str,self.items)) + "}"
    def __getitem__(self, index):
        return self.items.__getitem__(index)
    def __iter__(self):
       return self.items.__iter__() 
    def __len__(self):
       return self.items.__len__() 

class ListBundle(Bundle):
    """
    A bundle allocated to a single agent; contains a list of the item indices of names.
    """
    def __init__(self, items):
        if items is None:
            items = []
        self.items = sorted(items)

def bundle_from(b:Any):
    if isinstance(b,Bundle):
        return b
    elif b is None:
        return ListBundle([])
    elif isinstance(b,Iterable):
        return ListBundle(b)
    else:
        raise TypeError(f"Unsupported bundle type {type(b)}")
    
if __name__ == "__main__":
    import doctest
    (failures, tests) = doctest.testmod(report=True)
    print(f"{failures} failures, {tests} tests")