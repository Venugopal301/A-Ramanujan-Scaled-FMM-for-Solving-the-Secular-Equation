# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 10:22:18 2026

@author: se22pmat003
"""

import numpy as np


def multipole_to_local(r, a, b, dx, dy, scaling):
    """
    Multipole-to-local translation matrices.

    Parameters
    ----------
    r : int
        Expansion order.
    a : float
        Source box center.
    b : float
        Target box center.
    dx : float
        Source box diameter.
    dy : float
        Target box diameter.
    scaling : int
        0 -> unscaled
        1 -> Stirling scaled

    Returns
    -------
    B1 : ndarray (r,r)
        Translation matrix for 1/(x-y)

    B2 : ndarray (r,r)
        Translation matrix for 1/(x-y)^2
    """

    ab = a - b
    DD = np.diag((-1.0) ** np.arange(r))
    B1 = np.zeros((r, r))

    # =====================================================
    # B1
    # =====================================================

    if scaling == 0:

        cc = -np.ones(r) / ab

        for k in range(r - 1):
            cc[k + 1] = (k + 1) * cc[k] / ab

        for i in range(r):
            B1[i, : r - i] = cc[i:]

    elif scaling == 1:

        sr = (2 * np.pi * r) ** (1 / (2 * r)) / np.exp(1)

        B1[0, 0] = -1 / ab
        B1[0, 1] = -1 / ab**2 / (sr * 2 / dy)
        B1[1, 0] = -1 / ab**2 / (sr * 2 / dx)
        B1[1, 1] = 2 / ab * B1[1, 0] / (sr * 2 / dy)

        # -----------------------------------------
        # First row
        # -----------------------------------------

        for i in range(3, r + 1):
            B1[0, i - 1] = (
                (i - 1)
                / ab
                * B1[0, i - 2]
                / (sr * 2 / dy)
                / (i - 1)
                * (1 - 1 / (i - 1)) ** (i - 2)
            )
        # -----------------------------------------
        # Second row
        # -----------------------------------------
        for i in range(3, r):
            B1[1, i - 1] = (
                i / ab * B1[1, i - 2] / (sr * 2 / dy) / (i - 1) * (1 - 1 / (i - 1)) ** (i - 2)
            )

        # -----------------------------------------
        # First column
        # -----------------------------------------

        for i in range(3, r + 1):
            B1[i - 1, 0] = (
                (i - 1)
                / ab
                * B1[i - 2, 0]
                / (sr * 2 / dx)
                / (i - 1)
                * (1 - 1 / (i - 1)) ** (i - 2)
            )

        # -----------------------------------------
        # Second column
        # -----------------------------------------

        for i in range(3, r):
            B1[i - 1, 1] = (
                i / ab * B1[i - 2, 1] / (sr * 2 / dx) / (i - 1) * (1 - 1 / (i - 1)) ** (i - 2)
            )

        # -----------------------------------------
        # Interior
        # -----------------------------------------

        for k in range(3, r - 1):
            for i in range(3, r - k + 2):
                B1[k - 1, i - 1] = (
                    (k + i - 2)
                    / ab
                    * B1[k - 1, i - 2]
                    / (sr * 2 / dy)
                    / (i - 1)
                    * (1 - 1 / (i - 1)) ** (i - 2)
                )

    B1 = DD @ B1

    # =====================================================
    # B2
    # =====================================================

    B2 = np.zeros((r, r))

    if scaling == 0:

        eta = 1.0

        etaba = 1.0 / (eta * ab)

        cc = np.ones(r) / ab**2

        for k in range(r - 1):
            cc[k + 1] = (k + 2) * cc[k] * etaba

        for i in range(r):
            B2[i, : r - i] = cc[i:]

        B2 = DD @ B2

    elif scaling == 1:

        sr = (2 * np.pi * r) ** (1 / (2 * r)) / np.exp(1)
        B2[0, 0] = 1 / ab**2

        B2[0, 1] = 2 / ab**3 / (sr * 2 / dy)
        B2[1, 0] = 2 / ab**3 / (sr * 2 / dx)
        B2[1, 1] = 3 / ab * B2[1, 0] / (sr * 2 / dy)

        # -----------------------------------------
        # First row
        # -----------------------------------------

        for i in range(3, r + 1):
            B2[0, i - 1] = (
                i / ab * B2[0, i - 2] / (sr * 2 / dy) / (i - 1) * (1 - 1 / (i - 1)) ** (i - 2)
            )

        # -----------------------------------------
        # Second row
        # -----------------------------------------

        for i in range(3, r):
            B2[1, i - 1] = (
                (i + 1)
                / ab
                * B2[1, i - 2]
                / (sr * 2 / dy)
                / (i - 1)
                * (1 - 1 / (i - 1)) ** (i - 2)
            )

        # -----------------------------------------
        # First column
        # -----------------------------------------

        for i in range(3, r + 1):
            B2[i - 1, 0] = (
                i / ab * B2[i - 2, 0] / (sr * 2 / dx) / (i - 1) * (1 - 1 / (i - 1)) ** (i - 2)
            )

        # -----------------------------------------
        # Second column
        # -----------------------------------------

        for i in range(3, r):
            B2[i - 1, 1] = (
                (i + 1)
                / ab
                * B2[i - 2, 1]
                / (sr * 2 / dx)
                / (i - 1)
                * (1 - 1 / (i - 1)) ** (i - 2)
            )

        # -----------------------------------------
        # Interior
        # -----------------------------------------

        for k in range(3, r - 1):
            for i in range(3, r - k + 2):
                B2[k - 1, i - 1] = (
                    (k + i - 1)
                    / ab
                    * B2[k - 1, i - 2]
                    / (sr * 2 / dy)
                    / (i - 1)
                    * (1 - 1 / (i - 1)) ** (i - 2)
                )

        B2 = B2 @ DD

    return B1, B2
