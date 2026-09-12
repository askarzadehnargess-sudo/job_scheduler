"""
scalability.py
--------------
Scalability analysis module for the Job Scheduling Optimization project.
Part of: MTO Manufacturing Floor Scheduling System

Tests all three algorithms across seven dataset sizes to show how
execution time and solution quality change as the problem scales up.

Dataset sizes tested: 10, 25, 50, 75, 100, 150, 200 jobs

This module answers three key questions:
    1. At what problem size does DP become computationally infeasible?
    2. How does Greedy execution time grow with dataset size?
    3. Does Greedy consistently outperform EDF at all scales?

Key findings from Day 7:
    - DP is practical (guaranteed optimal) up to ~50 jobs
    - DP hits the time ceiling at 75 jobs and stays there
    - Greedy grows near-linearly: 0.048ms (10 jobs) to 0.942ms (200 jobs)
    - EDF outperformed Greedy at 75 jobs — the only exception in 7 sizes
    - DP time grows exponentially: 0.4ms → 21ms → 1,188ms (10 → 25 → 50 jobs)

Generates two charts saved to report/figures/:
    - scalability_time.png   — line chart of execution time vs jobs (log scale)
    - scalability_profit.png — line chart of profit vs jobs

Usage:
    python -m benchmark.scalability
"""

import time
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
matplotlib.use('Agg')  # non-interactive backend — saves files without opening windows
import matplotlib.pyplot as plt

from utils.data_loader import load_jobs
from algorithms.greedy import greedy_schedule
from algorithms.edf import edf_schedule
from algorithms.dp import dp_schedule
from data_generator import generate_scalability_dataset


# ── configuration ─────────────────────────────────────────────────────────────

# Seven dataset sizes — chosen to show the growth curve clearly
SIZES = [10, 25, 50, 75, 100, 150, 200]

# Map sizes to their dataset files
# Sizes 10, 50, 200 reuse the original benchmark datasets (unchanged from Day 2)
# Sizes 25, 75, 100, 150 use independently generated scalability datasets
SIZE_FILES = {
    10:  "data/jobs_small.json",   # hand-crafted Day 1 dataset
    50:  "data/jobs_medium.json",  # Day 2 generated dataset
    200: "data/jobs_large.json",   # Day 2 generated dataset
}

FIGURES_DIR = "report/figures"

# Consistent colors matching benchmark.py
COLORS = {
    "Greedy": "#2196F3",   # blue
    "EDF":    "#FF9800",   # orange
    "DP":     "#4CAF50",   # green
}


# ── data preparation ───────────────────────────────────────────────────────────

def prepare_datasets():
    """
    Ensures all required dataset files exist before running analysis.

    For sizes 10, 50, 200: uses the original benchmark datasets.
    For sizes 25, 75, 100, 150: generates new scalability datasets
    using generate_scalability_dataset() with independent random seeds
    (seed = 42 + num_jobs) — these do not affect the original datasets.

    Populates the SIZE_FILES dictionary with all file paths.
    """
    for size in SIZES:
        if size not in SIZE_FILES:
            filepath = f"data/jobs_n{size}.json"
            # Only generate if the file doesn't already exist
            if not os.path.exists(filepath):
                generate_scalability_dataset(size)
            SIZE_FILES[size] = filepath

    print("All datasets ready.")


# ── scalability runner ─────────────────────────────────────────────────────────

def run_scalability():
    """
    Runs all three algorithms on all seven dataset sizes.

    For each size records:
        - Execution time in milliseconds (single run — no averaging)
        - Total profit achieved (euros)
        - Whether DP reached guaranteed optimal

    DP uses a shorter time limit than the main benchmark:
        - 10 seconds for sizes up to 50 jobs (reaches optimal)
        - 5 seconds for sizes 75+ (hits limit — returns best found)

    This keeps total runtime manageable (~40 seconds total).

    Returns:
        data (dict): results keyed by algorithm name, then metric
            data["Greedy"] = {
                "times":   [t_10, t_25, t_50, ...],   # ms per size
                "profits": [p_10, p_25, p_50, ...]    # euros per size
            }
            data["DP"] also has "optimal": [True, True, False, ...]
    """
    prepare_datasets()

    # Initialize results structure
    data = {
        "Greedy": {"times": [], "profits": []},
        "EDF":    {"times": [], "profits": []},
        "DP":     {"times": [], "profits": [], "optimal": []},
    }

    print("\n" + "="*65)
    print("  SCALABILITY ANALYSIS")
    print("="*65)
    print(f"  {'Jobs':>6} | {'Greedy':>12} | {'EDF':>12} | {'DP':>16}")
    print(f"  {'-'*60}")

    for size in SIZES:
        filepath = SIZE_FILES[size]
        jobs, _ = load_jobs(filepath)

        # ── Greedy ──
        start = time.perf_counter()
        _, p_g, _ = greedy_schedule(jobs)
        t_g = (time.perf_counter() - start) * 1000  # ms
        data["Greedy"]["times"].append(t_g)
        data["Greedy"]["profits"].append(p_g)

        # ── EDF ──
        start = time.perf_counter()
        _, p_e, _ = edf_schedule(jobs)
        t_e = (time.perf_counter() - start) * 1000  # ms
        data["EDF"]["times"].append(t_e)
        data["EDF"]["profits"].append(p_e)

        # ── DP — shorter timeout to keep total runtime manageable ──
        # Up to 50 jobs: 10 second limit (reaches guaranteed optimal)
        # 75+ jobs: 5 second limit (hits limit, returns best found)
        timeout = 10 if size <= 50 else 5
        start = time.perf_counter()
        _, p_d, _, is_opt = dp_schedule(jobs, time_limit_seconds=timeout)
        t_d = (time.perf_counter() - start) * 1000  # ms
        data["DP"]["times"].append(t_d)
        data["DP"]["profits"].append(p_d)
        data["DP"]["optimal"].append(is_opt)

        # ✓ = guaranteed optimal   ~ = time limit reached
        opt_marker = "✓" if is_opt else "~"
        print(f"  {size:>6} | {t_g:>9.3f}ms | {t_e:>9.3f}ms | "
              f"{t_d:>10.1f}ms {opt_marker} | "
              f"G:€{p_g} E:€{p_e} D:€{p_d}")

    print(f"  {'='*60}")
    print("  ✓ = DP guaranteed optimal   ~ = DP time-limited")

    return data


# ── chart generators ───────────────────────────────────────────────────────────

def chart_scalability_time(data):
    """
    Line chart showing execution time vs number of jobs (log scale).

    The logarithmic Y-axis is essential — without it, DP's steep growth
    would make Greedy and EDF lines invisible at the bottom of the chart.

    The chart visually demonstrates NP-hardness: DP's steep curve
    contrasted against Greedy and EDF's near-flat lines shows the
    fundamental difference between exponential and polynomial growth.

    Parameters:
        data (dict): results from run_scalability()
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo in ["Greedy", "EDF", "DP"]:
        ax.plot(SIZES, data[algo]["times"],
                marker='o', linewidth=2.5, markersize=7,
                label=algo, color=COLORS[algo])

        # Annotate the final data point for quick reading
        last_time = data[algo]["times"][-1]
        label = (f"{last_time:.1f}ms" if last_time < 1000
                 else f"{last_time/1000:.1f}s")
        ax.annotate(label,
                    xy=(SIZES[-1], last_time),
                    xytext=(8, 0), textcoords="offset points",
                    va='center', fontsize=9,
                    color=COLORS[algo], fontweight='bold')

    ax.set_yscale("log")  # log scale — essential for showing exponential growth
    ax.set_xlabel("Number of Jobs", fontsize=12)
    ax.set_ylabel("Execution Time (ms) — log scale", fontsize=12)
    ax.set_title("Scalability Analysis — Execution Time vs Problem Size\n"
                 "MTO Manufacturing Floor Scheduling",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(SIZES)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "scalability_time.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Chart saved: {path}")


def chart_scalability_profit(data):
    """
    Line chart showing total profit achieved vs number of jobs.

    Unlike the time chart, profit lines fluctuate rather than growing
    smoothly — each dataset size uses different randomly generated data,
    so the job mix and difficulty varies at each size point. This is
    expected and scientifically honest behavior with synthetic datasets.

    The general upward trend shows that more jobs = more opportunities
    to fill the 8-hour shift with high-value combinations.

    Parameters:
        data (dict): results from run_scalability()
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for algo in ["Greedy", "EDF", "DP"]:
        ax.plot(SIZES, data[algo]["profits"],
                marker='o', linewidth=2.5, markersize=7,
                label=algo, color=COLORS[algo])

    ax.set_xlabel("Number of Jobs", fontsize=12)
    ax.set_ylabel("Total Profit Achieved (€)", fontsize=12)
    ax.set_title("Scalability Analysis — Profit vs Problem Size\n"
                 "MTO Manufacturing Floor Scheduling",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(SIZES)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"€{v:,.0f}"))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "scalability_profit.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved: {path}")


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run the full scalability analysis and generate both charts
    # Usage: python -m benchmark.scalability
    os.makedirs(FIGURES_DIR, exist_ok=True)

    data = run_scalability()

    print("\n  Generating scalability charts...")
    chart_scalability_time(data)
    chart_scalability_profit(data)

    print("\n  Scalability analysis complete ✅")