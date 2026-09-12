"""
edf.py
------
Earliest Deadline First (EDF) algorithm for the Weighted Job Scheduling Problem.
Part of: MTO Manufacturing Floor Scheduling System

Strategy:
    Sort jobs by deadline in ascending order (earliest deadline first).
    Schedule jobs using fixed-point iteration — keep making passes through
    the list until no more jobs can be added in a complete pass.

Mathematical basis:
    Sort key: d_i (deadline, ascending)
    Sort order: d_1 <= d_2 <= d_3 <= ... <= d_n
    Schedule job i if: T_current + t_i <= d_i  AND  all deps in D_i scheduled

Relationship to Greedy:
    EDF and Greedy use identical algorithmic skeletons.
    The ONLY difference is the sort key:
        Greedy: sort by p_i / t_i  (profit density, descending)
        EDF:    sort by d_i        (deadline, ascending)
    This single difference produced a 14,500 euro gap on the large dataset.

Theoretical note:
    EDF is provably OPTIMAL for minimizing the number of late jobs
    (Moore's Algorithm, 1968) but NOT for maximizing profit.
    It treats all jobs as equally valuable — only caring about urgency.
    This is why it underperforms Greedy on profit-maximization instances.
    Used here as a BASELINE to show what happens when the wrong objective
    function is used for sorting.

Exception found in experiments:
    EDF outperformed Greedy on the 75-job dataset (26,100 vs 24,300 euros),
    demonstrating that profit-density ranking is not universally superior.

Time complexity:  O(n^2) worst case (n passes * n jobs per pass)
                  O(n log n) in practice (sort dominates, few passes needed)
Space complexity: O(n)

Optimality:       NOT guaranteed for profit maximization.
                  Optimality gap: 44.2% on small dataset, 29.7% on medium.
"""

from utils.data_loader import load_jobs


def edf_schedule(jobs):
    """
    Schedules jobs using Earliest Deadline First with fixed-point iteration.

    Core strategy: always prioritize the job with the nearest deadline.
    Uses fixed-point iteration to handle dependency constraints correctly.

    Note: EDF optimizes for urgency, not value. A job worth 700 euros with
    an early deadline is scheduled before a job worth 4,700 euros with a
    later deadline. This is the fundamental reason EDF underperforms Greedy
    on profit-maximization problems.

    Parameters:
        jobs (list): list of job dictionaries from load_jobs(), each with
                     keys: job_id, name, duration, deadline, profit, dependencies

    Returns:
        scheduled (list): list of scheduled job dictionaries in execution order
        total_profit (int): total profit earned from scheduled jobs (euros)
        current_time (int): total machine time used (hours)
    """

    # Step 1 — sort jobs by deadline ascending (earliest deadline first)
    # This is the only mathematical difference from the Greedy algorithm.
    # Greedy sorts by profit/duration ratio; EDF sorts by deadline value.
    sorted_jobs = sorted(
        jobs,
        key=lambda job: job["deadline"]  # smallest deadline = highest priority
    )

    scheduled = []        # jobs committed to scheduling
    total_profit = 0      # running total of earned profit
    current_time = 0      # machine clock — tracks current time in shift
    scheduled_ids = set() # set of job_ids already scheduled (fast lookup)

    # Step 2 — fixed-point iteration
    # Repeat passes until an entire pass adds no new jobs.
    # Handles dependencies: a job blocked in pass 1 may become
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


if __name__ == "__main__":
    # Run directly to test on a specific dataset
    # Usage: python -m algorithms.edf [small|medium|large]
    import sys
    dataset = sys.argv[1] if len(sys.argv) > 1 else "small"
    jobs, metadata = load_jobs(f"data/jobs_{dataset}.json")
    from algorithms.greedy import print_schedule
    scheduled, total_profit, current_time = edf_schedule(jobs)
    print_schedule(scheduled, total_profit, current_time, f"EDF — {dataset}")