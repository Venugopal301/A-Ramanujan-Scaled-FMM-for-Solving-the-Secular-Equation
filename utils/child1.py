# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 12:03:17 2026

@author: se22pmat003
"""


def child1(tr):
    """
    Construct children lists from parent array.

    Parameters
    ----------
    tr : array-like
        Parent array.

    Returns
    -------
    ch : list
        ch[i] contains children of node i+1
    """

    n = len(tr)

    ch = [[] for _ in range(n)]
    for i in range(1, n):

        parent = int(tr[i])

        if parent > 0:
            ch[parent - 1].append(i + 1)

    return ch
