# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 17:10:44 2026

@author: se22pmat003
"""

import numpy as np


def particle_to_local(x, r, a, dx, scaling):
    """
    Parameters
    ----------
    x : ndarray
        Source points.
    r : int
        Expansion order.
    a : float
        Center of cluster.
    dx : float
        Diameter of cluster.
    scaling : int
        1 -> Stirling scaling
        0 -> No scaling

    Returns
    -------
    U : ndarray, shape (n,r)
    """

    # Ensure x is a column vector
    x = np.asarray(x).reshape(-1)

    n = len(x)

    U = np.ones((n, r))

    sr = (2 * np.pi * r) ** (1 / (2 * r)) / np.exp(1)

    U[:, 1] = sr * 2.0 / dx * (x - a)

    if scaling == 1:
        for k in range(2, r):
            U[:, k] = (1 + 1 / (k - 1)) ** (k - 1) * sr * 2.0 / dx * (x - a) * U[:, k - 1]

    elif scaling == 0:
        U = np.ones((n, r))
        for k in range(1, r):
            U[:, k] = (1.0 / k) * (x - a) * U[:, k - 1]

    return U
