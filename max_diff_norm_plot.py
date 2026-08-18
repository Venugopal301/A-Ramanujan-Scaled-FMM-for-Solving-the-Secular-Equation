#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 18:53:30 2026

@author: venugopal
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from Stirling_scaling2 import Stirling_scaling2
from Rmnj_scaling2 import Rmnj_scaling2

# Parameters
scaling = 1
n_iter = 10  # Newton iterations

r_values = np.arange(5, 71, 5)

max_B1_difference = []

n_values = np.arange(5000, 5001, 5000)

rng = np.random.default_rng(12345)

for jj, n in enumerate(n_values):
    
    print(f"jj = {jj}, n = {n}")

    # Generate 2n random points
    z = rng.uniform(0.0, 1.0, 2 * n)
    z.sort()

    # Interlaced x and y
    x = z[0::2]
    y = z[1::2]
    q = rng.uniform(0.5, 1.0, n)
    q = q**2  # elementwise square
    
    for r in r_values:

        print(f"r = {r}")
    
        # Stirling scaling
        interaction_pairs_S, B1_norms_S, B2_norms_S = Stirling_scaling2(
            r, x, y, q, scaling
        )
    
        # Ramanujan scaling
        interaction_pairs_R, B1_norms_R, B2_norms_R = Rmnj_scaling2(
            r, x, y, q, scaling
        )
    
        # Convert B1 norms to arrays
        B1_norms_R = np.asarray(B1_norms_R)
        B1_norms_S = np.asarray(B1_norms_S)
    
        # Absolute difference for every interaction
        difference_B1 = B1_norms_S - B1_norms_R
    
        # Maximum absolute difference
        max_diff = np.max(difference_B1)
    
        max_B1_difference.append(max_diff)
    
        print(f"Maximum |B1_S - B1_R| = {max_diff:.6e}")
        

results_table = pd.DataFrame({
    "r": r_values,
    "Maximum B1 Norm Difference": max_B1_difference
})

print(results_table.to_string(index=False))    

plt.figure(figsize=(8, 5))

plt.plot(
    r_values,
    max_B1_difference,
    marker='o'
)

plt.xlabel("Expansion order ($r$)")
plt.ylabel(r"$\max (\|B_1\|_S-\|B_1\|_R$)")
plt.title(r"Maximum difference in M2L Matrix $1$-Norms ($n=5000$)")
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()    