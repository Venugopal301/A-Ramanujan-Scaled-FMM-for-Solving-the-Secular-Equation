# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 15:38:07 2026

@author: se22pmat003
"""


def sib(tr, ch, i):
    """
    Return the sibling of node i.

    Parameters
    ----------
    tr : array-like
        Parent array.
    ch : list
        Child list.
    i : int

    Returns
    -------
    s : int
        Sibling node number.
    """

    parent = int(tr[i - 1])

    children = ch[parent - 1]

    if i == children[0]:
        s = children[1]
    else:
        s = children[0]

    return s
