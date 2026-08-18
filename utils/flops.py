# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 16:53:33 2026

@author: se22pmat003
"""

# import numpy as np


def flops(op_type, A, transA=None, B=None, transB=None, *args):
    """
    Parameters
    ----------
    op_type : str
        'prod', 'prodsym', 'sum', 'sumsym',
        'mv', 'rdiv', 'ldiv', 'chol',
        'lu', 'qr', 'ldl'

    A, B : numpy arrays
    transA, transB : str
        'n', 't', 'tri', etc.

    *args :
        Additional matrices/transposes for nested products.

    Returns
    -------
    nflops : float
    """

    # --------------------------------------------------
    # Matrix-Matrix Product
    # --------------------------------------------------
    if op_type == "prod":

        if transA.lower() == "n":
            m, n = A.shape
        else:
            n, m = A.shape

        if B.ndim == 1:
            p = 1
        else:
            if transB.lower() == "n":
                p = B.shape[1]
            else:
                p = B.shape[0]

        nflops = float(m * (2 * n - 1) * p)

        if len(args) > 0:

            for k in range(0, len(args), 2):

                M = args[k]
                transM = args[k + 1]

                m = p

                if transM.lower() == "n":
                    n = M.shape[0]
                    p = M.shape[1]
                else:
                    n = M.shape[1]
                    p = M.shape[0]

                nflops += float(m * (2 * n - 1) * p)

        return nflops

    # --------------------------------------------------
    # Symmetric Product
    # --------------------------------------------------
    if op_type == "prodsym":

        m, n = A.shape

        if transA.lower() == "n":
            nflops = float((2 * n - 1) * m * (m + 1) / 2)
        else:
            nflops = float((2 * m - 1) * n * (n + 1) / 2)

        return nflops

    # --------------------------------------------------
    # Sum
    # --------------------------------------------------
    if op_type == "sum":
        return float(A.size)

    # --------------------------------------------------
    # Symmetric Sum
    # --------------------------------------------------
    if op_type == "sumsym":
        return float(A.size / 2 + A.shape[0] / 2)

    # --------------------------------------------------
    # Matrix-Vector Product
    # --------------------------------------------------
    if op_type == "mv":

        if len(args) > 0:

            nflops = 0.0

            all_args = [A, transA, B, transB] + list(args)

            for k in range(len(all_args) // 2 - 1, -1, -1):

                M = all_args[2 * k]
                transM = all_args[2 * k + 1]

                if transM.lower() == "n":
                    m, n = M.shape
                else:
                    n, m = M.shape

                nflops += float(m * (2 * n - 1))

            return nflops

        else:

            if transA.lower() == "n":
                m, n = A.shape
            else:
                n, m = A.shape

            return float(m * (2 * n - 1))

    # --------------------------------------------------
    # Right Division
    # --------------------------------------------------
    if op_type == "rdiv":

        if transA.lower() == "n":

            if transB == "tri":
                nflops = float(A.shape[0] * B.shape[0] ** 2)
            else:
                nflops = float(A.shape[0] * (2 / 3) * B.shape[0] ** 3)

        else:

            if transB == "tri":
                nflops = float(A.shape[1] * B.shape[0] ** 2)
            else:
                nflops = float(2 * A.shape[1] * B.shape[0] ** 2 + (2 / 3) * B.shape[0] ** 3)

        return nflops

    # --------------------------------------------------
    # Left Division
    # --------------------------------------------------
    if op_type == "ldiv":

        if transB.lower() == "n":

            if transA == "tri":
                nflops = float(B.shape[1] * A.shape[0] ** 2)
            else:
                nflops = float(B.shape[1] * (2 / 3) * A.shape[0] ** 3)

        else:

            if transA == "tri":
                nflops = float(B.shape[0] * A.shape[0] ** 2)
            else:
                nflops = float(2 * B.shape[0] * A.shape[0] ** 2 + (2 / 3) * A.shape[0] ** 3)

        return nflops

    # --------------------------------------------------
    # Cholesky
    # --------------------------------------------------
    if op_type == "chol":

        n = A.shape[0]

        return float(n**3 / 3 - n**2 + 2 * n / 3)

    # --------------------------------------------------
    # LU
    # --------------------------------------------------
    if op_type == "lu":

        n = A.shape[0]

        return float(2 * n**3 / 3 - n**2 / 2 - n / 6)

    # --------------------------------------------------
    # QR
    # --------------------------------------------------
    if op_type == "qr":

        m, n = A.shape

        return float(2 * m * n**2 - (2 / 3) * n**3)

    # --------------------------------------------------
    # LDL
    # --------------------------------------------------
    if op_type == "ldl":

        n = A.shape[0]

        return float(n**3 / 3 - n**2 + 2 * n / 3)

    raise ValueError(f"Unknown operation type: {op_type}")
