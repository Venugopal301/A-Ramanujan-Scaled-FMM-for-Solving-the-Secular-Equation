# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 11:25:05 2026

@author: se22pmat003
"""

import numpy as np


def push(S, U):
    """
    Parameters
    ----------
    S : list
        Stack/list.
    U : ndarray

    Returns
    -------
    S : list
    """

    U = np.asarray(U)

    nr, nc = U.shape

    item = np.concatenate(
        (
            np.array([nr, nc]),
            U.reshape(-1, order="F"),
        )
    )

    S.append(item)

    return S
