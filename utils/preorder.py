# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 12:14:12 2026

@author: se22pmat003
"""


def preorder(i, tr, ch):
    """
    Return the preorder traversal of the subtree rooted at node i.

    Parameters
    ----------
    i : int
    tr : array-like
        Parent array (included for compatibility).
    ch : list
        Children list from child1().

    Returns
    -------
    list
        Preorder traversal.
    """

    children = ch[i - 1]

    if len(children) == 0:
        return [i]

    c1 = children[0]
    c2 = children[1]

    pre_ord_c1 = preorder(c1, tr, ch)
    pre_ord_c2 = preorder(c2, tr, ch)

    return [i] + pre_ord_c1 + pre_ord_c2
