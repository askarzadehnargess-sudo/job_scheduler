"""
main.py
-------
Main entry point for the Job Scheduling Optimization project.
Part of: MTO Manufacturing Floor Scheduling System

================================================================================
PROJECT OVERVIEW
================================================================================

Problem:
    In a Make to Order (MTO) manufacturing environment, a factory receives
    customer orders throughout the day. Each order (job) has a profit, a
    delivery deadline, a processing duration, and sometimes depends on other
    jobs being completed first. The single bottleneck machine can only process
    one job at a time. The goal is to select and sequence jobs to maximize
    total profit while respecting all deadlines and dependencies.

Formal definition:
    Given n jobs J = {j1, j2, ..., jn}, each with:
        pi  = profit (euros)
        di  = deadline (hours from shift start)
        ti  = duration (hours)
        Di  = set of prerequisite job_ids

    Find subset S ⊆ J that maximizes: Σ pi for all ji in S
    Subject to:
        1. Deadline:     start(ji) + ti ≤ di          for all ji in S
        2. Single machine: no two jobs overlap in time
        3. Dependencies: finish(jk) ≤ start(ji)       for all jk in Di

Constraints modeled:
    - Single machine (one CNC/welding workstation — the bottleneck)
    - No preemption (jobs run to completion once started)
    - Binary selection (jobs are fully scheduled or skipped)
    - 8-hour shift horizon

Algorithms compared:
    1. Dynamic Programming (DP)  — guaranteed optimal, exponential time (NP-hard)
    2. Greedy                    — near-optimal, O(n log n) time
    3. Earliest Deadline First   — baseline, O(n log n) time

Datasets:
    Small  (10 jobs)  → data/jobs_small.json   [hand-crafted]
    Medium (50 jobs)  → data/jobs_medium.json  [generated, seed=42]
    Large  (200 jobs) → data/jobs_large.json   [generated, seed=42]

================================================================================
HOW TO RUN
================================================================================

Run all algorithms on one dataset:
    python main.py [small|medium|large]
    python main.py small     ← default if no argument given

Run individual algorithms:
    python -m algorithms.greedy [small|medium|large]
    python -m algorithms.edf   [small|medium|large]
    python -m algorithms.dp    [small|medium|large]

Run full benchmark (all algorithms, all datasets, generates charts):
    python -m benchmark.benchmark

Run scalability analysis (7 dataset sizes, 2 charts):
    python -m benchmark.scalability

Run unit tests:
    python -m tests.test_algorithms

Regenerate medium and large datasets:
    python data_generator.py

================================================================================
"""

import sys
import os


def main():
    """
    Runs all three scheduling algorithms on a chosen dataset and
    prints a side-by-side comparison of their results.

    Usage:
        python main.py [small|medium|large]

    Default dataset: small
    """
    # Choose dataset from command line argument
    dataset = sys.argv[1] if len(sys.argv) > 1 else "small"

    # Validate argument
    valid = ["small", "medium", "large"]
    if dataset not in valid:
        print(f"Unknown dataset '{dataset}'. Choose from: {valid}")
        sys.exit(1)

    filepath = f"data/jobs_{dataset}.json"

    # Import here to keep top of file clean
    from utils.data_loader import load_jobs
    from algorithms.greedy import greedy_schedule, print_schedule
    from algorithms.edf import edf_schedule
    from algorithms.dp import dp_schedule
    import time

    print("\n" + "="*65)
    print("  JOB SCHEDULING OPTIMIZATION — MTO Manufacturing Floor")
    print("="*65)
    print(f"  Dataset: {dataset} ({filepath})")
    print("="*65)

    # Load data
    jobs, metadata = load_jobs(filepath)
    print(f"  Shift duration: {metadata['shift_duration_hours']} hours")
    print(f"  Jobs available: {len(jobs)}")
    print()

    # ── Run Greedy ──
    start = time.perf_counter()
    s_g, p_g, t_g = greedy_schedule(jobs)
    time_g = (time.perf_counter() - start) * 1000
    print_schedule(s_g, p_g, t_g, f"Greedy (time: {time_g:.3f}ms)")

    # ── Run EDF ──
    start = time.perf_counter()
    s_e, p_e, t_e = edf_schedule(jobs)
    time_e = (time.perf_counter() - start) * 1000
    print_schedule(s_e, p_e, t_e, f"EDF (time: {time_e:.3f}ms)")

    # ── Run DP ──
    print("  Running DP (may take up to 30s for large dataset)...")
    start = time.perf_counter()
    s_d, p_d, t_d, is_optimal = dp_schedule(jobs)
    time_d = (time.perf_counter() - start) * 1000
    status = "GUARANTEED OPTIMAL" if is_optimal else "BEST FOUND (time-limited)"
    print_schedule(s_d, p_d, t_d, f"DP — {status} (time: {time_d:.1f}ms)")

    # ── Summary comparison ──
    print("="*65)
    print("  COMPARISON SUMMARY")
    print("="*65)
    print(f"  {'Algorithm':<12} {'Profit':>10} {'Time':>12} {'Jobs':>6}")
    print(f"  {'-'*45}")
    print(f"  {'Greedy':<12} €{p_g:>9,} {time_g:>10.3f}ms {len(s_g):>6}")
    print(f"  {'EDF':<12} €{p_e:>9,} {time_e:>10.3f}ms {len(s_e):>6}")
    print(f"  {'DP':<12} €{p_d:>9,} {time_d:>10.1f}ms {len(s_d):>6}  [{status}]")
    print(f"  {'-'*45}")

    # Optimality gap (only if DP is guaranteed optimal)
    if is_optimal and p_d > 0:
        gap_g = (p_d - p_g) / p_d * 100
        gap_e = (p_d - p_e) / p_d * 100
        print(f"\n  Optimality gaps vs DP:")
        print(f"    Greedy gap: {gap_g:.1f}%")
        print(f"    EDF gap:    {gap_e:.1f}%")

    print("="*65)
    print()
    print("  To run full benchmark:    python -m benchmark.benchmark")
    print("  To run scalability:       python -m benchmark.scalability")
    print("  To run tests:             python -m tests.test_algorithms")
    print()


if __name__ == "__main__":
    main()