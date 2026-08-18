# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 10:32:01 2026

@author: se22pmat003
"""

import numpy as np
import time
import pandas as pd

from Rmnj_scaling import Rmnj_scaling
from P2L import P2L
from flops import flops

# Preallocate containers
setup_time = np.zeros((10, 1))
elapsed_time = [[None for _ in range(2)] for _ in range(10)]
fmm_fun_val = [None for _ in range(10)]
fmm_y = [None for _ in range(10)]
flop_counts = [None for _ in range(10)]

exact_fun_val = [None] * 10
exact_y = [None] * 10

x_values = []

fmm_failures = []
exact_failures = []

# Parameters
scaling = 1
n_iter = 10  # Newton iterations
r = 20

n_values = np.arange(5000, 50001, 5000)

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
    x_values.append(x.copy())
    
    start_time = time.perf_counter()

    u1, u2, post_ord, ch, Px, Py, I, nflops, neighbor = Rmnj_scaling(r, x, y, q, scaling)
    
    setup_time[jj] = time.perf_counter() - start_time

    fmm_timer = np.zeros(n_iter)
    exact_timer = np.zeros(n_iter)

    for iter in range(n_iter):

        ftt = time.perf_counter()

        z1 = np.zeros(len(q))
        z2 = np.zeros(len(q))

        # ==========================================
        # Near-field interaction & final evaluation
        # ==========================================

        for i in post_ord:
            idx = i - 1
            if len(ch[idx]) == 0:
                py = Py[idx]
                if len(py) > 0:
                    yj = y[py]
                    if u1[idx] is not None:
                        # Ui = particle_to_local(yj, r, I[idx, 0], 2 * I[idx, 1], scaling)
                        Ui = P2L(yj, r, I[idx, 0], 2 * I[idx, 1], scaling)
                        z1[py] = Ui @ u1[idx]
                        z2[py] = Ui @ u2[idx]
                        nflops += flops("prod", Ui, "n", u1[idx], "n")
                        near_nodes = [i] + neighbor[idx]

                    for j in np.unique(near_nodes):
                        jdx = j - 1
                        px = Px[jdx]
                        if len(px) > 0:
                            xi = x[px]
                            D1 = 1.0 / (xi[np.newaxis, :] - yj[:, np.newaxis])
                            D2 = 1.0 / (xi[np.newaxis, :] - yj[:, np.newaxis]) ** 2
                            D1[np.isinf(D1)] = 0.0
                            D2[np.isinf(D2)] = 0.0
                            z1[py] += D1 @ q[px]
                            z2[py] += D2 @ q[px]
                            nflops += D1.shape[0] * (2 * D1.shape[1] - 1) + len(py)

        w1 = 1.0 + z1
        ynew = y - (w1 / z2)
        y = ynew

        fmm_timer[iter] = time.perf_counter() - ftt

    fmm_fun_val[jj] = w1.copy()  # store function values
    fmm_y[jj] = ynew.copy()  # store updated targets

    elapsed_time[jj][0] = fmm_timer  # first column

    flop_counts[jj] = nflops


# ==========================================
# Direct evaluation of secular equation
# ==========================================

    y = z[1::2]  # using y from the original data
    
    for iter in range(n_iter):

        dtt = time.perf_counter()

        ne = len(x)

        exact_fun = np.zeros(ne, dtype=np.float64)
        exact_der = np.zeros(ne, dtype=np.float64)

        for k in range(ne):

            dn = x - y[k]  # (x-y)
            der_dn = dn**2  # (x-y)^2

            exact_fun[k] = np.sum(q / dn)

            exact_der[k] = np.sum(q / der_dn)

        v1 = 1.0 + exact_fun

        ynew = y - (v1 / exact_der)

        y = ynew.copy()

        exact_timer[iter] = time.perf_counter() - dtt

    # ======================================
    # Store results
    # ======================================

    exact_fun_val[jj] = v1.copy()

    exact_y[jj] = ynew.copy()

    elapsed_time[jj][1] = exact_timer
    


#==============================================================
#========================ANALYSE RESULTS =======================
#===============================================================

# =====================SPEEDUP FACTORS =========================

elapsed_time = np.array(elapsed_time)
for i in range(10):

    # Timing data for the i-th dataset
    fmm_time = elapsed_time[i, 0, :]       # 10 Newton iterations
    direct_time = elapsed_time[i, 1, :]    # 10 Newton iterations

    # Speedup for each Newton iteration
    speedup = direct_time / fmm_time

    # Average speedup over the 10 Newton iterations
    average_speedup = np.mean(speedup)

    table = pd.DataFrame({
        "Newton Iteration": np.arange(1, 11),
        "FMM Time (s)": fmm_time,
        "Direct Time (s)": direct_time,
        "Speedup": speedup
    })

    print(f"\nDataset n = {n_values[i]}")
    print(table.to_string(index=False))

    print(f"Average speedup = {average_speedup:.4f}x")

#=======================================================================
#===================PRODUCE TIMING GRAPHS ===============================
#=======================================================================

# =========================================================
# Figure 1: n = 5000, 10000, 15000
# =========================================================

import matplotlib.pyplot as plt

# elapsed_time shape: (10, 2, 10)
# axis 0 -> data size
# axis 1 -> 0: FMM, 1: Direct
# axis 2 -> Newton iteration

n_values = np.arange(5000, 50001, 5000)
iterations = np.arange(1, 11)

# ---------------------------------------------------------
# Graph 1: n = 5000, 10000, 15000
# ---------------------------------------------------------

plt.figure(figsize=(9, 6))

for i in range(3):
    plt.plot(
        iterations,
        elapsed_time[i, 0, :],
        marker='o',
        label=f'FMM, n={n_values[i]}'
    )

    plt.plot(
        iterations,
        elapsed_time[i, 1, :],
        marker='s',
        linestyle='--',
        label=f'Direct, n={n_values[i]}'
    )

plt.xlabel('Newton iteration')
plt.ylabel('Time (s)')
plt.title('FMM vs Direct: n = 5000–15000')
plt.xticks(iterations)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# Graph 2: n = 45000, 50000
# ---------------------------------------------------------

plt.figure(figsize=(9, 6))

for i in range(8, 10):
    plt.plot(
        iterations,
        elapsed_time[i, 0, :],
        marker='o',
        label=f'FMM, n={n_values[i]}'
    )

    plt.plot(
        iterations,
        elapsed_time[i, 1, :],
        marker='s',
        linestyle='--',
        label=f'Direct, n={n_values[i]}'
    )

plt.xlabel('Newton iteration')
plt.ylabel('Time (s)')
plt.title('FMM vs Direct: n = 45000–50000')
plt.xticks(iterations)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


#========================HISTOGRAMS FOR n = 5000 AND 50000 ===================

# ==================================================
# Magnitude bins
# ==================================================

bins = np.concatenate((10.0**np.arange(-13, 0), [np.inf]))

labels = [
    r"$10^{-13}$-$10^{-12}$",
    r"$10^{-12}$-$10^{-11}$",
    r"$10^{-11}$-$10^{-10}$",
    r"$10^{-10}$-$10^{-9}$",
    r"$10^{-9}$-$10^{-8}$",
    r"$10^{-8}$-$10^{-7}$",
    r"$10^{-7}$-$10^{-6}$",
    r"$10^{-6}$-$10^{-5}$",
    r"$10^{-5}$-$10^{-4}$",
    r"$10^{-4}$-$10^{-3}$",
    r"$10^{-3}$-$10^{-2}$",
    r"$10^{-2}$-$10^{-1}$",
    r"$>10^{-1}$"
]


# ==================================================
# Function to plot histogram
# ==================================================

def plot_residual_histogram(fmm_fun_val, exact_fun_val, index, n):

    # Absolute function values
    fmm_vals = np.abs(fmm_fun_val[index])
    exact_vals = np.abs(exact_fun_val[index])

    # Histogram counts
    fmm_counts, _ = np.histogram(fmm_vals, bins=bins)
    exact_counts, _ = np.histogram(exact_vals, bins=bins)

    # Print counts
    print(f"\nAbsolute function value distribution for n = {n}")
    print("Magnitude Bin".ljust(25), "FMM".rjust(8), "Exact".rjust(8))
    print("-" * 45)

    for lab, fc, ec in zip(labels, fmm_counts, exact_counts):
        print(f"{lab:<25}{fc:>8}{ec:>8}")

    # ==================================================
    # Grouped bar chart
    # ==================================================

    x = np.arange(len(labels))
    width = 0.38

    plt.figure(figsize=(12, 6))

    bars1 = plt.bar(
        x - width/2,
        fmm_counts,
        width,
        label="FMM"
    )

    bars2 = plt.bar(
        x + width/2,
        exact_counts,
        width,
        label="Exact"
    )

    # Counts above FMM bars
    for bar in bars1:
        h = bar.get_height()

        if h > 0:
            plt.text(
                bar.get_x() + bar.get_width()/2,
                h,
                f"{int(h)}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    # Counts above Exact bars
    for bar in bars2:
        h = bar.get_height()

        if h > 0:
            plt.text(
                bar.get_x() + bar.get_width()/2,
                h,
                f"{int(h)}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    plt.xticks(
        x,
        labels,
        rotation=45,
        ha="right"
    )

    plt.xlabel(r"Absolute function value $|f(y)|$")
    plt.ylabel("Number of Roots")
    plt.title(rf"Absolute function value distribution for $n={n}$, $r=20$")

    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.show()


# ==================================================
# n = 5000
# ==================================================

plot_residual_histogram(
    fmm_fun_val,
    exact_fun_val,
    index=0,
    n=5000
)


# ==================================================
# n = 50000
# ==================================================

plot_residual_histogram(
    fmm_fun_val,
    exact_fun_val,
    index=9,
    n=50000
)

#============================================================
#================ INTERLACE FAILURE COUNTS ==================
#===========================================================

def count_interlacing_failures(x, y):

    x = np.asarray(x)
    y = np.asarray(y)

    failures = np.sum(
        (y[:-1] <= x[:-1]) |
        (y[:-1] >= x[1:])
    )

    failures += int(y[-1] <= x[-1])

    return int(failures)


fmm_failures = []
exact_failures = []

for i, n in enumerate(n_values):

    fmm_failures.append(
        count_interlacing_failures(x_values[i], fmm_y[i])
    )

    exact_failures.append(
        count_interlacing_failures(x_values[i], exact_y[i])
    )


# Table
interlacing_table = pd.DataFrame({
    "n": n_values,
    "FMM": fmm_failures,
    "Direct": exact_failures
})

print("\nInterlacing failures:")
print(interlacing_table.to_string(index=False))


#=========================================================================
#========================FLOP COUNTS ====================================
#==========================================================================


flop_count = np.asarray(flop_counts)

plt.figure(figsize=(8, 5))

plt.plot(n_values, flop_count, marker='o')

plt.xlabel("Data size ($n$)")
plt.ylabel("FLOP count")
plt.title("FLOP Count vs. Data Size")

plt.xticks(n_values)
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

