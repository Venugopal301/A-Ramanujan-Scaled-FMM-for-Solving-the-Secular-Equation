# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 15:08:33 2026

@author: se22pmat003
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 17:14:43 2026

@author: se22pmat003
"""

import numpy as np
from math import factorial


def M2M(r, a, b, dx, dy, scaling):
    """
    Compute M2M translation matrix.

    Parameters
    ----------
    r : int
        Expansion order.
    a : float
        Source cluster center.
    b : float
        Target cluster center.
    dx : float
        Diameter of source cluster.
    dy : float
        Diameter of target cluster.
    scaling : int
        0 = unscaled
        1 = Stirling-scaled
        2 = explicit scaling matrices

    Returns
    -------
    T : ndarray (r,r)
    """

    sr = (np.sqrt(np.pi) * (8 * r**3 + 4 * r**2 + r + 1 / 30) ** (1 / 6)) ** (
        1 / r
    ) / np.exp(1)

    # ----------------------------------
    # Initialize diagonal
    # ----------------------------------

    diag_entries = (dx / dy) ** np.arange(r)

    T = np.diag(diag_entries)

    ab = a - b

    T[0, 1] = ab * (sr * 2.0 / dy)

    # ----------------------------------
    # Consistency check
    # ----------------------------------

    k = 1
    i = 2

    exact = (
        (i - 1) ** (i - 1)
        / (sr ** (k - i))
        * (dx / 2) ** (k - 1)
        / (dy / 2) ** (i - 1)
        * ab ** (i - k)
        / factorial(i - k)
    )

    if abs(T[k - 1, i - 1] - exact) > 1e-16:
        print("Consistency check failed")
        raise ValueError("Error in M2M initialization")

    # ----------------------------------
    # Scaling mode 1
    # ----------------------------------

    if scaling == 1:

        #
        # First row
        #
        for i in range(3, r + 1):

            T[0, i - 1] = (
                ab
                / (i - 1)
                * T[0, i - 2]
                * (sr * 2.0 / dy)
                * (i - 1)
                / (1 - 1 / (i - 1)) ** (i - 2)
            )

            k = 1

            exact = (
                (i - 1) ** (i - 1)
                * sr ** (i - k)
                * (dx / 2) ** (k - 1)
                / (dy / 2) ** (i - 1)
                * ab ** (i - k)
                / factorial(i - k)
            )

            if abs(T[k - 1, i - 1]) > 1:
                print("T =", T[k - 1, i - 1])
                print("difference =", abs(T[k - 1, i - 1] - exact))
                raise ValueError("Bound violation")

        #
        # Remaining rows
        #
        for k in range(2, r + 1):

            for i in range(k + 1, r + 1):

                T[k - 1, i - 1] = (
                    ab
                    / (i - k)
                    * T[k - 1, i - 2]
                    * (sr * 2.0 / dy)
                    * (i - 1)
                    / (1 - 1 / (i - 1)) ** (i - 2)
                )

    # ----------------------------------
    # Scaling mode 0
    # ----------------------------------

    elif scaling == 0:

        cc = np.ones(r)

        for k in range(r - 1):

            cc[k + 1] = ab * cc[k] / (k + 1)

        T = np.zeros((r, r))

        for i in range(r):

            T[i, i:] = cc[: r - i]

    # ----------------------------------
    # Scaling mode 2
    # ----------------------------------

    elif scaling == 2:

        ma = sr / dx * 2

        Sa = np.diag(np.concatenate(([1], (ma * np.arange(1, r)) ** np.arange(1, r))))

        mb = sr / dy * 2

        Sb = np.diag(np.concatenate(([1], (mb * np.arange(1, r)) ** np.arange(1, r))))

        cc = np.ones(r)

        for k in range(r - 1):

            cc[k + 1] = ab * cc[k] / (k + 1)

        T = np.zeros((r, r))

        for i in range(r):

            T[i, i:] = cc[: r - i]

        T = np.linalg.inv(Sa) @ T @ Sb

    return T
