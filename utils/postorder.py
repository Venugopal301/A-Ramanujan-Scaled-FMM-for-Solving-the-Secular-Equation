# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 12:11:08 2026

@author: se22pmat003
"""


def postorder(i, tr, ch):
    """
    Return the postorder traversal of the subtree rooted at node i.

    Parameters
    ----------
    i : int
    tr : array-like
        Parent array (not actually used here, but kept for compatibility).
    ch : list
        Children list from child1().

    Returns
    -------
    list
        Postorder traversal.
    """

    # print("Entering node:", i)

    children = ch[i - 1]

    # print("children =", children)

    if len(children) == 0:
        return [i]

    c1 = children[0]
    c2 = children[1]

    post_ord_c1 = postorder(c1, tr, ch)
    post_ord_c2 = postorder(c2, tr, ch)

    return post_ord_c1 + post_ord_c2 + [i]
