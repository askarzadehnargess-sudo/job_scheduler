"""
data_loader.py
--------------
Data loading and validation module for the Job Scheduling Optimization project.
Part of: MTO Manufacturing Floor Scheduling System

Responsibilities:
    - Read job datasets from JSON files
    - Validate all required fields are present
    - Validate all values are logically consistent
    - Provide a debug-friendly job printer

Used by: greedy.py, edf.py, dp.py, benchmark.py, scalability.py, test_algorithms.py
"""

import json
import os


def load_jobs(filepath):
    """
    Reads a JSON file and returns the list of jobs and metadata.

    Validates that:
        - The file exists
        - All required fields are present in every job
        - All values are logically consistent (no negative durations, etc.)
        - No job has a duration that exceeds its own deadline

    Parameters:
        filepath (str): path to the JSON file (e.g. 'data/jobs_small.json')

    Returns:
        jobs (list): list of job dictionaries, each with keys:
                     job_id, name, duration, deadline, profit, dependencies
        metadata (dict): dataset info (scenario, shift_duration_hours, num_jobs, etc.)

    Raises:
        FileNotFoundError: if the file does not exist at the given path
        ValueError: if any job is missing a required field or has invalid values
    """

    # Step 1 — check the file actually exists before trying to open it
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    # Step 2 — open and parse the JSON file into Python objects
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Step 3 — extract the jobs list and dataset metadata
    jobs = data["jobs"]
    metadata = data["metadata"]

    # Step 4 — validate every job has all required fields
    required_fields = ["job_id", "name", "duration", "deadline",
                       "profit", "dependencies"]
    for job in jobs:
        for field in required_fields:
            if field not in job:
                raise ValueError(
                    f"Job {job.get('job_id', '?')} is missing field: {field}"
                )

    # Step 5 — validate all values are logically consistent
    for job in jobs:
        if job["duration"] <= 0:
            raise ValueError(
                f"Job {job['job_id']} has invalid duration: {job['duration']}"
            )
        if job["deadline"] <= 0:
            raise ValueError(
                f"Job {job['job_id']} has invalid deadline: {job['deadline']}"
            )
        if job["profit"] <= 0:
            raise ValueError(
                f"Job {job['job_id']} has invalid profit: {job['profit']}"
            )
        if job["duration"] > job["deadline"]:
            raise ValueError(
                f"Job {job['job_id']} duration ({job['duration']}h) exceeds "
                f"its deadline ({job['deadline']}h) — impossible to complete!"
            )

    print(f"Loaded {len(jobs)} jobs from {filepath}")
    return jobs, metadata


def print_jobs(jobs):
    """
    Prints a formatted table of all jobs to the terminal.

    Useful for debugging and visual inspection of a dataset.
    Shows job_id, name, duration, deadline, profit, and dependencies
    in a clean aligned table format.

    Parameters:
        jobs (list): list of job dictionaries from load_jobs()
    """
    print(f"\n{'ID':<8} {'Name':<25} {'Dur':>4} {'Deadline':>9} "
          f"{'Profit':>8}  Dependencies")
    print("-" * 72)
    for job in jobs:
        deps = ", ".join(job["dependencies"]) if job["dependencies"] else "—"
        print(f"{job['job_id']:<8} {job['name']:<25} {job['duration']:>4}h "
              f"{job['deadline']:>8}h  €{job['profit']:>6}  {deps}")
    print()