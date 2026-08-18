#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 15:12:19 2026

@author: venugopal
"""

import numpy as np
import matplotlib.pyplot as plt

from Stirling_scaling2 import Stirling_scaling2
from Rmnj_scaling2 import Rmnj_scaling2

# Parameters
scaling = 1
n_iter = 10  # Newton iterations

r = 50

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

    interaction_pairs_S, B1_norms_S, B2_norms_S = Stirling_scaling2(r, x, y, q, scaling)
    interaction_pairs_R, B1_norms_R, B2_norms_R = Rmnj_scaling2(r, x, y, q, scaling)



    # Convert lists to arrays
    B1_norms_R = np.asarray(B1_norms_R)
    B1_norms_S = np.asarray(B1_norms_S)
    B2_norms_R = np.asarray(B2_norms_R)
    B2_norms_S = np.asarray(B2_norms_S)
    
    # Interaction index
    k = np.arange(1, len(B1_norms_R) + 1)
    
    
    #====================SCATTER PLOT USING LOG SCALE ==============================
    plt.figure(figsize=(8, 5))
    
    difference_B1 =  np.asarray(B1_norms_S) - np.asarray(B1_norms_R)
    
    plt.scatter(k, difference_B1, s=15)
    
    plt.yscale("log")
    
    plt.xlabel("Interaction index")
    plt.ylabel(r"$\|B_1\|_S-\|B_1\|_R$")
    plt.title(f"Difference in M2L Matrix Norms (n = {n}, r = {r})")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
    
    
    