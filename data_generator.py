"""
data_generator.py
-----------------
Synthetic dataset generator for the Job Scheduling Optimization project.
Part of: MTO Manufacturing Floor Scheduling System

Generates realistic job datasets for an MTO (Make to Order) manufacturing
floor scenario. All parameters are grounded in realistic manufacturing logic.

Parameter justification:
    Duration:        1-4 hours — typical machining/welding task durations
    Deadline:        duration+1 to 8 hours — same-shift delivery commitment
    Profit:          500-5000 euros — realistic MTO order revenue range
    Dependency rate: ~30% — reflects real production chains without
                     over-constraining the scheduling problem

Two types of datasets are generated:

    Main datasets (fixed seed=42, used for all algorithm comparisons):
        - jobs_small.json  — 10 jobs (hand-crafted on Day 1, not generated)
        - jobs_medium.json — 50 jobs
        - jobs_large.json  — 200 jobs

    Scalability datasets (seed=42+size, independent per size):
        - jobs_n25.json, jobs_n75.json, jobs_n100.json, jobs_n150.json
        - Used exclusively for Day 7 scalability analysis
        - Independent seeds ensure each size is reproducible on its own

Reproducibility:
    random.seed(42) is set for main datasets so all benchmarks always
    run on identical data. This is standard OR research practice.

Usage:
    python data_generator.py              # regenerate medium + large datasets
    from data_generator import generate_scalability_dataset
    generate_scalability_dataset(75)      # generate a 75-job scalability dataset
"""

import json
import random
import os

# ── constants ──────────────────────────────────────────────────────────────────

# Fix the random seed for reproducibility of main datasets.
# All benchmarks must run on identical data across all runs.
random.seed(42)

# Realistic MTO manufacturing job names — 30 templates
# For datasets larger than 30 jobs, names cycle with a numeric suffix
# e.g. "Cut Steel Plates #2", "Cut Steel Plates #3", etc.
JOB_NAMES = [
    "Cut Steel Plates",       "Cut Aluminum Sheets",
    "Weld Frame Assembly",    "Weld Support Brackets",
    "Drill Mounting Holes",   "Drill Precision Holes",
    "Grind Surface Finish",   "Grind Edge Chamfers",
    "Paint Body Panel",       "Paint Frame Structure",
    "Assemble Main Unit",     "Assemble Sub-Components",
    "Install Fasteners",      "Install Hydraulic Lines",
    "Quality Inspection",     "Final Quality Test",
    "Package Assembly",       "Package Finished Unit",
    "Surface Treatment",      "Heat Treatment",
    "CNC Milling Operation",  "CNC Turning Operation",
    "Press Brake Forming",    "Plasma Cutting",
    "TIG Welding",            "MIG Welding",
    "Powder Coating",         "Anodizing Treatment",
    "Thread Tapping",         "Deburring Operation",
]


# ── core generators ────────────────────────────────────────────────────────────

def generate_job(job_index, all_previous_ids, dependency_rate=0.30):
    """
    Generates a single realistic MTO manufacturing job.

    Each job represents one customer order on the production floor.
    Parameters are randomly generated within realistic manufacturing ranges.

    Parameters:
        job_index (int): sequential job number — used to build job_id (J001, J002...)
        all_previous_ids (list): job_ids of all jobs generated before this one.
                                  Used to assign realistic dependencies.
        dependency_rate (float): probability this job depends on a prior job.
                                  Default 0.30 = 30% dependency rate.

    Returns:
        dict: job dictionary with keys:
              job_id, name, duration, deadline, profit, dependencies
    """

    # Job ID — zero-padded 3-digit number: J001, J002, ..., J200
    job_id = f"J{job_index:03d}"

    # Job name — cycle through JOB_NAMES list, add suffix after first cycle
    name_index = (job_index - 1) % len(JOB_NAMES)
    cycle      = (job_index - 1) // len(JOB_NAMES)
    name = JOB_NAMES[name_index]
    if cycle > 0:
        name = f"{name} #{cycle + 1}"

    # Duration: 1-4 hours — typical single-workstation task duration
    duration = random.randint(1, 4)

    # Deadline: duration+1 to 8 hours — job must be completable within shift
    # At least 1 hour of slack ensures every generated job is theoretically schedulable
    min_deadline = duration + 1
    max_deadline = 8
    if min_deadline > max_deadline:
        min_deadline = max_deadline  # safety for 4-hour jobs
    deadline = random.randint(min_deadline, max_deadline)

    # Profit: 500-5000 euros, rounded to nearest 100 for realism
    profit = random.randint(5, 50) * 100

    # Dependencies: 30% chance of depending on one randomly chosen prior job
    # Dependencies only point backwards (to lower job indices) — no cycles possible
    dependencies = []
    if all_previous_ids and random.random() < dependency_rate:
        dep = random.choice(all_previous_ids)
        dependencies = [dep]

    return {
        "job_id":       job_id,
        "name":         name,
        "duration":     duration,
        "deadline":     deadline,
        "profit":       profit,
        "dependencies": dependencies
    }


def generate_dataset(num_jobs, filename, dataset_size_label):
    """
    Generates a full dataset and saves it to a JSON file.

    Uses the global random seed (42) set at module level. This means
    the medium and large datasets are always identical across runs —
    essential for reproducible benchmarking.

    Parameters:
        num_jobs (int): number of jobs to generate
        filename (str): output file path (e.g. 'data/jobs_medium.json')
        dataset_size_label (str): human-readable label ('medium', 'large')
    """
    jobs = []
    all_ids = []

    for i in range(1, num_jobs + 1):
        job = generate_job(i, all_ids)
        jobs.append(job)
        all_ids.append(job["job_id"])

    dataset = {
        "metadata": {
            "scenario":             "MTO Manufacturing Floor",
            "shift_duration_hours": 8,
            "num_jobs":             num_jobs,
            "dataset_size":         dataset_size_label,
            "random_seed":          42,
            "dependency_rate":      0.30
        },
        "jobs": jobs
    }

    os.makedirs("data", exist_ok=True)

    with open(filename, 'w') as f:
        json.dump(dataset, f, indent=2)

    # Print generation summary
    deps_count   = sum(1 for j in jobs if j["dependencies"])
    avg_profit   = sum(j["profit"]   for j in jobs) / num_jobs
    avg_duration = sum(j["duration"] for j in jobs) / num_jobs

    print(f"Generated {num_jobs} jobs -> {filename}")
    print(f"  Dependencies: {deps_count} ({deps_count/num_jobs*100:.0f}%) | "
          f"Avg profit: €{avg_profit:.0f} | Avg duration: {avg_duration:.1f}h")
    print()


def generate_scalability_dataset(num_jobs):
    """
    Generates a dataset for scalability analysis with an independent random seed.

    Uses seed = 42 + num_jobs so each dataset size is independently
    reproducible without affecting the main datasets (medium, large)
    or other scalability datasets.

    This function was added on Day 7 without modifying generate_dataset()
    to preserve the integrity of all Day 3-6 benchmark results.

    Parameters:
        num_jobs (int): number of jobs to generate

    Returns:
        filepath (str): path to the generated JSON file
    """
    # Independent seed per size — does not affect global random state
    # for the main datasets
    random.seed(42 + num_jobs)

    jobs = []
    all_ids = []

    for i in range(1, num_jobs + 1):
        job = generate_job(i, all_ids)
        jobs.append(job)
        all_ids.append(job["job_id"])

    dataset = {
        "metadata": {
            "scenario":             "MTO Manufacturing Floor",
            "shift_duration_hours": 8,
            "num_jobs":             num_jobs,
            "dataset_size":         f"scalability_{num_jobs}",
            "random_seed":          42 + num_jobs,
            "dependency_rate":      0.30
        },
        "jobs": jobs
    }

    os.makedirs("data", exist_ok=True)
    filepath = f"data/jobs_n{num_jobs}.json"

    with open(filepath, 'w') as f:
        json.dump(dataset, f, indent=2)

    return filepath


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Regenerates the medium and large datasets.
    # WARNING: running this overwrites jobs_medium.json and jobs_large.json.
    # The small dataset (jobs_small.json) was hand-crafted on Day 1
    # and is never regenerated by this script.
    print("Generating main datasets for Job Scheduling Optimization")
    print("=" * 55)

    generate_dataset(50,  "data/jobs_medium.json", "medium")
    generate_dataset(200, "data/jobs_large.json",  "large")

    print("All datasets generated successfully!")
    print("Files saved in the data/ folder.")