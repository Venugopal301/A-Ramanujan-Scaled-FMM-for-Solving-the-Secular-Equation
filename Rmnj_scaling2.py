# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 15:09:37 2026

@author: se22pmat003
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 10:55:19 2026

@author: se22pmat003
"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))


import numpy as np
from child1 import child1
from postorder import postorder
from preorder import preorder
from sib import sib
from P2L import P2L
from M2L import M2L
from M2M import M2M
from flops import flops


def Rmnj_scaling2(r, x, y, q, scaling):

    N0 = 64
    tau = 0.6

    z2 = max(np.max(x), np.max(y))
    z1 = min(np.min(x), np.min(y))

    z2 += 0.1 * abs(z2)
    z1 -= 0.1 * abs(z1)

    nflops = 0

    # Root box
    I = np.array([[(z1 + z2) / 2, (z2 - z1) / 2, 1, 1, 1, 0, 1]], dtype=float)

    S = []

    S.append(I)

    Px = [np.arange(len(x))]
    Py = [np.arange(len(y))]

    while len(S) > 0:

        i = S.pop().ravel()

        c = i[0]
        d = i[1]
        el = int(i[2])
        er = int(i[3])
        idx = int(i[4])
        lvl = int(i[6])

        n = I.shape[0]

        px = Px[idx - 1]
        Px[idx - 1] = np.array([], dtype=int)

        py = Py[idx - 1]
        Py[idx - 1] = np.array([], dtype=int)

        # -----------------------------------
        # Bisect interval
        # -----------------------------------
        i1 = np.array([[c - d / 2, d / 2, el, 1, n + 1, idx, lvl + 1]])
        i2 = np.array([[c + d / 2, d / 2, 0, er, n + 2, idx, lvl + 1]])

        I = np.vstack((I, i1, i2))

        # -----------------------------------
        # Left child
        # -----------------------------------

        if el == 1:

            Px.append(px[(c - d <= x[px]) & (x[px] <= c)])

            Py.append(py[(c - d <= y[py]) & (y[py] <= c)])

        else:

            Px.append(px[(c - d < x[px]) & (x[px] <= c)])

            Py.append(py[(c - d < y[py]) & (y[py] <= c)])

        # -----------------------------------
        # Right child
        # -----------------------------------

        if er == 1:

            Px.append(px[(c < x[px]) & (x[px] <= c + d)])

            Py.append(py[(c < y[py]) & (y[py] <= c + d)])

        else:

            Px.append(px[(c < x[px]) & (x[px] < c + d)])

            Py.append(py[(c < y[py]) & (y[py] < c + d)])

        # -----------------------------------
        # Adaptive subdivision
        # -----------------------------------
        left_idx = len(Px) - 2
        right_idx = len(Px) - 1

        if len(Px[left_idx]) > N0 or len(Py[left_idx]) > N0:
            S.append(i1)

        if len(Px[right_idx]) > N0 or len(Py[right_idx]) > N0:
            S.append(i2)

    # Parent array
    tr = I[:, 5].astype(int)

    # Child array
    ch = child1(tr)

    # Number of nodes
    n = len(tr)

    # Traversals
    post_ord = postorder(1, tr, ch)
    pre_ord = preorder(1, tr, ch)

    # -----------------------------------
    # Neighbor and interaction lists
    # -----------------------------------

    neighbor = [[] for _ in range(n)]
    interlist = [[] for _ in range(n)]

    for i in pre_ord:
        parent = int(tr[i - 1])
        if parent == 0:
            neighbor[i - 1] = []
            interlist[i - 1] = []
        else:
            neighbor[i - 1] = [sib(tr, ch, i)]
            cousins = []
            for j in neighbor[parent - 1]:
                if len(ch[j - 1]) == 0:
                    cousins.append(j)
                else:
                    cousins.append(ch[j - 1][0])
                    cousins.append(ch[j - 1][1])
            ci = I[i - 1, 0]
            ri = I[i - 1, 1]
            for j in cousins:
                cj = I[j - 1, 0]
                rj = I[j - 1, 1]
                if ri + rj <= tau * abs(ci - cj):
                    interlist[i - 1].append(j)
                else:
                    neighbor[i - 1].append(j)

    for i in pre_ord:
        if len(ch[i - 1]) == 0:
            for j in neighbor[i - 1]:
                tmp = set(neighbor[j - 1])
                tmp.add(i)
                neighbor[j - 1] = list(tmp)

            for j in interlist[i - 1]:
                tmp = set(interlist[j - 1])
                tmp.add(i)
                interlist[j - 1] = list(tmp)

        u1 = [None] * n
        u2 = [None] * n
        v1 = [None] * n

        # ==========================================
        # Post-order (bottom-up) traversal
        # ==========================================

    for i in post_ord:
        idx = i - 1
        lvl = int(I[idx, 6])

        if lvl >= 2:
            if len(ch[idx]) == 0:
                xi = x[Px[idx]]
                Vi = P2L(xi, r, I[idx, 0], 2 * I[idx, 1], scaling)
                v1[idx] = Vi.T @ q[Px[idx]]
                nflops += Vi.shape[1] * (2 * Vi.shape[0] - 1)
            else:
                c1 = ch[idx][0]
                c2 = ch[idx][1]
                c1_idx = c1 - 1
                c2_idx = c2 - 1
                Wc1 = M2M(
                    r, I[c1_idx, 0], I[idx, 0], 2 * I[c1_idx, 1], 2 * I[idx, 1], scaling
                )
                Wc2 = M2M(
                    r, I[c2_idx, 0], I[idx, 0], 2 * I[c2_idx, 1], 2 * I[idx, 1], scaling
                )
                v1[idx] = Wc1.T @ v1[c1_idx] + Wc2.T @ v1[c2_idx]
                nflops += (
                    flops("prod", Wc1, "t", v1[c1_idx], "n")
                    + flops("prod", Wc2, "t", v1[c2_idx], "n")
                    + v1[idx].size
                )

            # ==========================================
            # Pre-order (top-down) traversal
            # ==========================================
    B1_norms = []
    B2_norms = []
    interaction_pairs = []
    
    for i in pre_ord:
        idx = i - 1
        lvl = int(I[idx, 6])
        if lvl >= 2:
            # ----------------------------------
            # M2L interactions
            # ----------------------------------

            interaction_list = interlist[idx]
            for j in interaction_list:
                j_idx = j - 1
                B1, B2 = M2L(
                    r, I[idx, 0], I[j_idx, 0], 2 * I[idx, 1], 2 * I[j_idx, 1], scaling
                )
                
                interaction_pairs.append([i, j])
                B1_norms.append(np.linalg.norm(B1,1))
                B2_norms.append(np.linalg.norm(B2,1))

                if u1[idx] is None:
                    u1[idx] = B1 @ v1[j_idx]
                    u2[idx] = B2 @ v1[j_idx]
                    nflops += flops("prod", B1, "n", v1[j_idx], "n")
                    nflops += flops("prod", B2, "n", v1[j_idx], "n")
                else:
                    u1[idx] += B1 @ v1[j_idx]
                    u2[idx] += B2 @ v1[j_idx]
                    nflops += flops("prod", B1, "n", v1[j_idx], "n") + u1[idx].size
                    nflops += flops("prod", B2, "n", v1[j_idx], "n") + u1[idx].size

            # ----------------------------------
            # Downward pass
            # ----------------------------------

            if lvl > 3:
                p = int(tr[idx])
                # if p != 0:
                p_idx = p - 1
                if u1[p_idx] is not None:
                    Ri = M2M(
                        r, I[idx, 0], I[p_idx, 0], 2 * I[idx, 1], 2 * I[p_idx, 1], scaling
                    )

                    if u1[idx] is None:
                        u1[idx] = Ri @ u1[p_idx]
                        u2[idx] = Ri @ u2[p_idx]
                        nflops += flops("prod", Ri, "n", u1[p_idx], "n")
                        nflops += flops("prod", Ri, "n", u2[p_idx], "n")
                    else:
                        u1[idx] += Ri @ u1[p_idx]
                        u2[idx] += Ri @ u2[p_idx]
                        nflops += flops("prod", Ri, "n", u1[p_idx], "n") + u1[idx].size
                        nflops += flops("prod", Ri, "n", u2[p_idx], "n") + u2[idx].size

    return (interaction_pairs, B1_norms, B2_norms)
