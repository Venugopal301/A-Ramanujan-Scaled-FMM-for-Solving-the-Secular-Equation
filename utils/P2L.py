# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 15:05:43 2026

@author: se22pmat003
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 17:10:44 2026

@author: se22pmat003
"""

import numpy as np


def P2L(x, r, a, dx, scaling):
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

    sr = (np.sqrt(np.pi) * (8 * r**3 + 4 * r**2 + r + 1 / 30) ** (1 / 6)) ** (
        1 / r
    ) / np.exp(1)

    U[:, 1] = sr * 2.0 / dx * (x - a)

    if scaling == 1:
        for k in range(2, r):
            U[:, k] = (1 + 1 / (k - 1)) ** (k - 1) * sr * 2.0 / dx * (x - a) * U[:, k - 1]

    elif scaling == 0:
        U = np.ones((n, r))
        for k in range(1, r):
            U[:, k] = (1.0 / k) * (x - a) * U[:, k - 1]

    return U
