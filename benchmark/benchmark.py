"""
benchmark.py
------------
Performance benchmarking module for the Job Scheduling Optimization project.
Part of: MTO Manufacturing Floor Scheduling System

Measures and compares all three scheduling algorithms across all three datasets:
    - Execution time (milliseconds) — averaged over 3 runs for accuracy
    - Total profit achieved (euros)
    - Optimality gap vs DP optimal (% difference from guaranteed best)

Generates three comparison charts saved to report/figures/:
    - profit_comparison.png  — grouped bar chart of profit per algorithm
    - execution_time.png     — grouped bar chart of time (log scale)
    - optimality_gap.png     — gap chart for small and medium datasets

Algorithms: Greedy, EDF, Dynamic Programming (DP)
Datasets:   Small (10 jobs), Medium (50 jobs), Large (200 jobs)

Usage:
    python -m benchmark.benchmark
"""

import time
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
matplotlib.use('Agg')  # non-interactive backend — saves files without opening windows
import matplotlib.pyplot as plt
import numpy as np  # used for bar chart x-axis positioning

from utils.data_loader import load_jobs
from algorithms.greedy import greedy_schedule
from algorithms.edf import edf_schedule
from algorithms.dp import dp_schedule


# ── configuration ─────────────────────────────────────────────────────────────

# Datasets to benchmark — (display_name, filepath, num_jobs)
DATASETS = [
    ("Small",  "data/jobs_small.json",  10),
    ("Medium", "data/jobs_medium.json", 50),
    ("Large",  "data/jobs_large.json",  200),
]

# Heuristic algorithms to benchmark (DP handled separately due to slow runtime)
ALGORITHMS = {
    "Greedy": greedy_schedule,
    "EDF":    edf_schedule,
}

FIGURES_DIR = "report/figures"   # output directory for charts
RESULTS_DIR = "benchmark/results" # output directory for raw results

# Consistent colors across all charts
COLORS = {
    "Greedy": "#2196F3",   # blue
    "EDF":    "#FF9800",   # orange
    "DP":     "#4CAF50",   # green
}


# ── timing helper ──────────────────────────────────────────────────────────────

def measure_time(fn, *args, repeats=3):
    """
    Measures the average execution time of a function call over multiple runs.

    Running multiple times and averaging reduces noise from system fluctuations
    (CPU scheduling, memory cache effects, background processes).

    Parameters:
        fn: the function to time
        *args: positional arguments to pass to fn
        repeats (int): number of times to run fn (default: 3)

    Returns:
        avg_time_ms (float): average execution time in milliseconds
        result: return value of the last function call
    """
    times = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()   # high-precision timer
        result = fn(*args)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # seconds → milliseconds

    avg_time_ms = sum(times) / len(times)
    return avg_time_ms, result


# ── benchmark runner ───────────────────────────────────────────────────────────

def run_benchmarks():
    """
    Runs all algorithms on all datasets and collects performance metrics.

    Greedy and EDF are each run 3 times and averaged.
    DP is run once only — it can take up to 30 seconds on large datasets.

    Optimality gap is computed for datasets where DP reaches guaranteed optimal:
        gap = (DP_profit - algo_profit) / DP_profit * 100

    Returns:
        results (dict): results for Greedy and EDF, keyed by dataset then algorithm
            results[dataset_name][algo_name] = {
                profit, time_ms, jobs_scheduled, time_used, optimality_gap
            }
        dp_results (dict): DP results keyed by dataset name
            dp_results[dataset_name] = {
                profit, time_ms, jobs_scheduled, time_used, is_optimal
            }
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = {}
    dp_results = {}

    print("\n" + "="*65)
    print("  BENCHMARKING — Job Scheduling Optimization")
    print("="*65)

    for dataset_name, filepath, num_jobs in DATASETS:
        print(f"\n  Dataset: {dataset_name} ({num_jobs} jobs)")
        print(f"  {'-'*55}")

        jobs, metadata = load_jobs(filepath)
        results[dataset_name] = {}

        # Run Greedy and EDF — 3 repeats each, take average
        for algo_name, algo_fn in ALGORITHMS.items():
            avg_time, result = measure_time(algo_fn, jobs)
            scheduled, profit, time_used = result

            results[dataset_name][algo_name] = {
                "profit":         profit,
                "time_ms":        avg_time,
                "jobs_scheduled": len(scheduled),
                "time_used":      time_used,
            }

            print(f"  {algo_name:<8} | profit=€{profit:<7} | "
                  f"time={avg_time:.3f}ms | jobs={len(scheduled)}")

        # Run DP once — can be slow, especially on large dataset
        print(f"  DP       | running... (may take up to 30s for large dataset)")
        dp_start = time.perf_counter()
        scheduled_dp, profit_dp, time_used_dp, is_optimal = dp_schedule(jobs)
        dp_elapsed = (time.perf_counter() - dp_start) * 1000

        dp_results[dataset_name] = {
            "profit":         profit_dp,
            "time_ms":        dp_elapsed,
            "jobs_scheduled": len(scheduled_dp),
            "time_used":      time_used_dp,
            "is_optimal":     is_optimal,
        }

        status = "OPTIMAL" if is_optimal else "TIME-LIMITED"
        print(f"  DP       | profit=€{profit_dp:<7} | "
              f"time={dp_elapsed:.1f}ms | jobs={len(scheduled_dp)} | {status}")

        # Compute optimality gap only where DP is guaranteed optimal
        if is_optimal and profit_dp > 0:
            for algo_name in ALGORITHMS:
                algo_profit = results[dataset_name][algo_name]["profit"]
                gap = (profit_dp - algo_profit) / profit_dp * 100
                results[dataset_name][algo_name]["optimality_gap"] = gap
                print(f"  {algo_name} optimality gap vs DP: {gap:.1f}%")
        else:
            # Gap is undefined when DP did not reach guaranteed optimal
            for algo_name in ALGORITHMS:
                results[dataset_name][algo_name]["optimality_gap"] = None

    return results, dp_results


# ── chart generators ───────────────────────────────────────────────────────────

def chart_profit_comparison(results, dp_results):
    """
    Grouped bar chart showing total profit per algorithm per dataset.

    Three bars per dataset group (Greedy, EDF, DP) with profit values
    labeled on top of each bar. Y-axis formatted in euros.

    Parameters:
        results (dict): Greedy and EDF results from run_benchmarks()
        dp_results (dict): DP results from run_benchmarks()
    """
    dataset_names = [d[0] for d in DATASETS]
    x = np.arange(len(dataset_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    greedy_profits = [results[d]["Greedy"]["profit"] for d in dataset_names]
    edf_profits    = [results[d]["EDF"]["profit"]    for d in dataset_names]
    dp_profits     = [dp_results[d]["profit"]        for d in dataset_names]

    bars1 = ax.bar(x - width, greedy_profits, width, label="Greedy",
                   color=COLORS["Greedy"], alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x,         edf_profits,   width, label="EDF",
                   color=COLORS["EDF"],    alpha=0.85, edgecolor="white")
    bars3 = ax.bar(x + width, dp_profits,    width, label="DP (optimal/best)",
                   color=COLORS["DP"],     alpha=0.85, edgecolor="white")

    # Label each bar with its profit value
    for bar in bars1 + bars2 + bars3:
        height = bar.get_height()
        ax.annotate(f"€{height:,}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xlabel("Dataset Size", fontsize=12)
    ax.set_ylabel("Total Profit (€)", fontsize=12)
    ax.set_title("Profit Comparison — Greedy vs EDF vs DP\n"
                 "MTO Manufacturing Floor Scheduling",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d[0]}\n({d[2]} jobs)" for d in DATASETS])
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"€{v:,.0f}"))
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(dp_profits) * 1.15)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "profit_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Chart saved: {path}")


def chart_execution_time(results, dp_results):
    """
    Grouped bar chart showing execution time per algorithm per dataset.

    Uses logarithmic Y-axis scale — essential because DP execution time
    is orders of magnitude larger than Greedy and EDF. Without log scale,
    Greedy and EDF bars would be invisible.

    Parameters:
        results (dict): Greedy and EDF results from run_benchmarks()
        dp_results (dict): DP results from run_benchmarks()
    """
    dataset_names = [d[0] for d in DATASETS]
    x = np.arange(len(dataset_names))
    width = 0.25

    greedy_times = [results[d]["Greedy"]["time_ms"] for d in dataset_names]
    edf_times    = [results[d]["EDF"]["time_ms"]    for d in dataset_names]
    dp_times     = [dp_results[d]["time_ms"]        for d in dataset_names]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width, greedy_times, width, label="Greedy",
                   color=COLORS["Greedy"], alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x,         edf_times,   width, label="EDF",
                   color=COLORS["EDF"],    alpha=0.85, edgecolor="white")
    bars3 = ax.bar(x + width, dp_times,    width, label="DP",
                   color=COLORS["DP"],     alpha=0.85, edgecolor="white")

    # Label Greedy and EDF bars in milliseconds
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}ms",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7)

    # Label DP bars — convert to seconds if over 1000ms
    for bar in bars3:
        height = bar.get_height()
        label = f"{height:.0f}ms" if height < 1000 else f"{height/1000:.1f}s"
        ax.annotate(label,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7)

    ax.set_yscale("log")  # log scale — makes Greedy/EDF visible alongside DP
    ax.set_xlabel("Dataset Size", fontsize=12)
    ax.set_ylabel("Execution Time (ms) — log scale", fontsize=12)
    ax.set_title("Execution Time Comparison — Greedy vs EDF vs DP\n"
                 "MTO Manufacturing Floor Scheduling",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d[0]}\n({d[2]} jobs)" for d in DATASETS])
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "execution_time.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved: {path}")


def chart_optimality_gap(results):
    """
    Bar chart showing optimality gap (%) for Greedy and EDF vs DP optimal.

    Only includes datasets where DP reached guaranteed optimal solution.
    Large dataset is excluded because DP timed out — gap is undefined.

    Lower gap = better algorithm performance relative to optimal.

    Parameters:
        results (dict): Greedy and EDF results from run_benchmarks()
                        Must include 'optimality_gap' key per algorithm
    """
    # Collect only datasets where DP was guaranteed optimal
    valid_datasets = []
    greedy_gaps = []
    edf_gaps = []

    for d_name, _, _ in DATASETS:
        g_gap = results[d_name]["Greedy"].get("optimality_gap")
        e_gap = results[d_name]["EDF"].get("optimality_gap")
        if g_gap is not None:
            valid_datasets.append(d_name)
            greedy_gaps.append(g_gap)
            edf_gaps.append(e_gap)

    if not valid_datasets:
        print("  No optimal DP results available for gap chart.")
        return

    x = np.arange(len(valid_datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))

    bars1 = ax.bar(x - width/2, greedy_gaps, width, label="Greedy gap",
                   color=COLORS["Greedy"], alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + width/2, edf_gaps,   width, label="EDF gap",
                   color=COLORS["EDF"],    alpha=0.85, edgecolor="white")

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlabel("Dataset Size", fontsize=12)
    ax.set_ylabel("Optimality Gap (%)", fontsize=12)
    ax.set_title("Optimality Gap vs DP Optimal\n"
                 "(lower = better; only shown where DP is guaranteed optimal)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(valid_datasets)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(greedy_gaps), max(edf_gaps)) * 1.2)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "optimality_gap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved: {path}")


def print_summary_table(results, dp_results):
    """
    Prints a formatted results summary table to the terminal.

    Shows profit, execution time, jobs scheduled, and optimality gap
    for all algorithms across all datasets in a single aligned table.

    Parameters:
        results (dict): Greedy and EDF results from run_benchmarks()
        dp_results (dict): DP results from run_benchmarks()
    """
    print(f"\n{'='*65}")
    print(f"  RESULTS SUMMARY TABLE")
    print(f"{'='*65}")
    print(f"  {'Dataset':<10} {'Algorithm':<10} {'Profit':>8} "
          f"{'Time':>12} {'Jobs':>6} {'Gap':>8}")
    print(f"  {'-'*60}")

    for d_name, _, n_jobs in DATASETS:
        for algo in ["Greedy", "EDF"]:
            r = results[d_name][algo]
            gap = r.get("optimality_gap")
            gap_str = f"{gap:.1f}%" if gap is not None else "N/A"
            print(f"  {d_name:<10} {algo:<10} €{r['profit']:>7,} "
                  f"{r['time_ms']:>10.3f}ms {r['jobs_scheduled']:>6} {gap_str:>8}")

        dp = dp_results[d_name]
        status = "OPTIMAL" if dp["is_optimal"] else "TIME-LIM"
        print(f"  {d_name:<10} {'DP':<10} €{dp['profit']:>7,} "
              f"{dp['time_ms']:>10.1f}ms {dp['jobs_scheduled']:>6} "
              f"{'—':>8} [{status}]")
        print(f"  {'-'*60}")

    print(f"{'='*65}\n")


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run the full benchmark and generate all charts
    # Usage: python -m benchmark.benchmark
    results, dp_results = run_benchmarks()
    print_summary_table(results, dp_results)

    print("\n  Generating charts...")
    chart_profit_comparison(results, dp_results)
    chart_execution_time(results, dp_results)
    chart_optimality_gap(results)

    print("\n  All charts saved to report/figures/")
    print("  Benchmark complete ✅\n")