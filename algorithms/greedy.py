"""
greedy.py
---------
Greedy algorithm for the Weighted Job Scheduling Problem.
Part of: MTO Manufacturing Floor Scheduling System

Strategy:
    Sort jobs by profit-to-duration ratio (profit density) in descending order.
    Schedule jobs using fixed-point iteration — keep making passes through the
    list until no more jobs can be added in a complete pass.

Mathematical basis:
    Profit density: rho_i = p_i / t_i  (euros per hour)
    Sort order: rho_1 >= rho_2 >= ... >= rho_n
    Schedule job i if: T_current + t_i <= d_i  AND  all deps in D_i scheduled

Correctness note:
    Fixed-point iteration (Day 5 fix) ensures jobs are reconsidered after
    their dependencies become available. Original one-pass approach permanently
    skipped dependency-blocked jobs even if prerequisites were scheduled later.

Time complexity:  O(n^2) worst case (n passes * n jobs per pass)
                  O(n log n) in practice (sort dominates, few passes needed)
Space complexity: O(n)

Optimality:       NOT guaranteed — locally optimal decisions can miss the
                  global optimum, especially in dependency-rich instances.
                  Optimality gap: 44.2% on small dataset, 9.9% on medium.
"""

from utils.data_loader import load_jobs


def greedy_schedule(jobs):
    """
    Schedules jobs using the Greedy algorithm with fixed-point iteration.

    Core strategy: rank jobs by profit density (rho = profit / duration).
    Higher profit per hour of machine time = higher priority.
    Uses fixed-point iteration to handle dependency constraints correctly.

    Parameters:
        jobs (list): list of job dictionaries from load_jobs(), each with
                     keys: job_id, name, duration, deadline, profit, dependencies

    Returns:
        scheduled (list): list of scheduled job dictionaries in execution order
        total_profit (int): total profit earned from scheduled jobs (euros)
        current_time (int): total machine time used (hours)
    """

    # Step 1 — sort jobs by profit density: rho_i = p_i / t_i (descending)
    # This is the mathematical core of the Greedy algorithm.
    # Jobs that earn more euros per hour of machine time are prioritized.
    sorted_jobs = sorted(
        jobs,
        key=lambda job: job["profit"] / job["duration"],
        reverse=True  # highest profit density first
    )

    scheduled = []       # jobs we have committed to scheduling
    total_profit = 0     # running total of earned profit
    current_time = 0     # machine clock — tracks current time in shift
    scheduled_ids = set() # set of job_ids already scheduled (fast lookup)

    # Step 2 — fixed-point iteration
    # Repeat passes until an entire pass adds no new jobs.
    # This handles dependencies: a job blocked in pass 1 may become
    # available in pass 2 after its prerequisite is scheduled.
    remaining_jobs = sorted_jobs.copy()
    made_progress = True

    while made_progress and remaining_jobs:
        made_progress = False
        still_remaining = []

        for job in remaining_jobs:

            # Condition 1 — all dependencies must already be scheduled
            # Formally: for all j_k in D_i, j_k must be in scheduled_ids
            deps_met = all(dep in scheduled_ids for dep in job["dependencies"])
            if not deps_met:
                still_remaining.append(job)
                continue

            # Condition 2 — job must finish at or before its deadline
            # Formally: T_current + t_i <= d_i
            finish_time = current_time + job["duration"]
            if finish_time <= job["deadline"]:
                # Both conditions met — schedule this job
                scheduled.append(job)
                scheduled_ids.add(job["job_id"])
                current_time = finish_time
                total_profit += job["profit"]
                made_progress = True  # trigger another pass
            else:
                # Job misses deadline at current time — cannot schedule
                still_remaining.append(job)

        remaining_jobs = still_remaining

    return scheduled, total_profit, current_time


def print_schedule(scheduled, total_profit, current_time, algorithm_name="Greedy"):
    """
    Prints a formatted timeline of a scheduled job sequence to the terminal.

    Displays each job as a time block showing start time, end time,
    job ID, job name, and profit. Prints a summary footer with totals.

    Parameters:
        scheduled (list): list of scheduled job dicts in execution order
        total_profit (int): total profit earned (euros)
        current_time (int): total machine time used (hours)
        algorithm_name (str): label shown in the header (default: "Greedy")
    """
    print(f"\n{'='*55}")
    print(f"  {algorithm_name} Schedule")
    print(f"{'='*55}")

    if not scheduled:
        print("  No jobs could be scheduled.")
    else:
        time = 0
        for job in scheduled:
            start = time
            end = time + job["duration"]
            print(f"  [{start}h -> {end}h]  "
                  f"{job['job_id']} {job['name']:<25} €{job['profit']}")
            time = end

    print(f"{'─'*55}")
    print(f"  Total jobs scheduled : {len(scheduled)}")
    print(f"  Total time used      : {current_time}h / 8h")
    print(f"  Total profit earned  : €{total_profit}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    # Run directly to test on a specific dataset
    # Usage: python -m algorithms.greedy [small|medium|large]
    import sys
    dataset = sys.argv[1] if len(sys.argv) > 1 else "small"
    jobs, metadata = load_jobs(f"data/jobs_{dataset}.json")
    scheduled, total_profit, current_time = greedy_schedule(jobs)
    print_schedule(scheduled, total_profit, current_time, f"Greedy — {dataset}")