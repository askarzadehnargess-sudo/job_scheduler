# Job Scheduling Optimization — MTO Manufacturing Floor


A Python system that solves the weighted job scheduling problem for a Make
to Order (MTO) manufacturing environment. Three algorithms are implemented,
compared, and benchmarked: Dynamic Programming (DP), Greedy, and Earliest
Deadline First (EDF).

---

## The Problem

In a Make to Order factory, each customer order (job) has:
- A profit — revenue earned if completed on time (euros)
- A deadline — must finish by this hour in the 8-hour shift
- A duration — how many hours it takes to complete
- Dependencies — some jobs cannot start until others finish

The single bottleneck machine can only run one job at a time.
The goal is to select and sequence jobs to maximize total profit
while respecting all deadlines and dependencies.

This is a precedence-constrained weighted job scheduling problem.
It is proven NP-hard when job dependencies are included.

---

## Algorithms

Three algorithms are implemented and compared:

1. Dynamic Programming (DP)
   - Exhaustive recursive search with memoization
   - Guaranteed optimal solution for small and medium instances
   - Time complexity: O(2^n) — exponential due to NP-hardness
   - Becomes infeasible beyond approximately 50 jobs

2. Greedy
   - Sorts jobs by profit-to-duration ratio (profit density)
   - Uses fixed-point iteration to handle dependencies
   - Time complexity: O(n log n)
   - Near-optimal heuristic — practical for any dataset size

3. Earliest Deadline First (EDF) — baseline
   - Sorts jobs by deadline (earliest first)
   - Same structure as Greedy but wrong objective for profit maximization
   - Time complexity: O(n log n)
   - Used as comparison baseline only

---

## Project Structure

    job_scheduler/
    |
    |-- data/
    |   |-- jobs_small.json          10 jobs, hand-crafted
    |   |-- jobs_medium.json         50 jobs, generated
    |   |-- jobs_large.json          200 jobs, generated
    |   |-- jobs_n25.json            25 jobs, scalability dataset
    |   |-- jobs_n75.json            75 jobs, scalability dataset
    |   |-- jobs_n100.json           100 jobs, scalability dataset
    |   |-- jobs_n150.json           150 jobs, scalability dataset
    |
    |-- algorithms/
    |   |-- greedy.py                Greedy algorithm
    |   |-- edf.py                   EDF baseline algorithm
    |   |-- dp.py                    Dynamic Programming algorithm
    |
    |-- utils/
    |   |-- data_loader.py           JSON reader and validator
    |
    |-- benchmark/
    |   |-- benchmark.py             Full benchmark — 3 algorithms x 3 datasets
    |   |-- scalability.py           Scalability analysis — 7 dataset sizes
    |
    |-- tests/
    |   |-- test_algorithms.py       8 unit tests + brute-force gold standard
    |
    |-- report/
    |   |-- figures/                 5 generated charts (PNG files)
    |       |-- profit_comparison.png
    |       |-- execution_time.png
    |       |-- optimality_gap.png
    |       |-- scalability_time.png
    |       |-- scalability_profit.png
    |
    |-- data_generator.py            Dataset generator
    |-- main.py                      Main entry point
    |-- README.md                    This file

---

## How to Run

Always run commands from the project root folder using Anaconda Prompt.

    cd C:\Users\zahra\Desktop\job_scheduler

Run all algorithms on one dataset:

    python main.py small
    python main.py medium
    python main.py large

Run a single algorithm:

    python -m algorithms.greedy small
    python -m algorithms.edf medium
    python -m algorithms.dp large

Run the full benchmark (measures time and profit, generates 3 charts):

    python -m benchmark.benchmark

Run scalability analysis (7 dataset sizes, generates 2 charts):

    python -m benchmark.scalability

Run unit tests:

    python -m tests.test_algorithms

Regenerate medium and large datasets:

    python data_generator.py

---

## Key Results

Profit comparison across all datasets and algorithms:

    Dataset          Greedy      EDF         DP          DP Status
    Small (10)       4,300       4,300       7,700       Guaranteed optimal
    Medium (50)      18,200      14,200      20,200      Guaranteed optimal
    Large (200)      33,500      19,000      29,900      Time-limited

Optimality gaps vs DP optimal:

    Small dataset:   Greedy 44.2%,  EDF 44.2%
    Medium dataset:  Greedy 9.9%,   EDF 29.7%
    Large dataset:   not calculable (DP did not reach guaranteed optimal)

Execution time comparison:

    Dataset          Greedy      EDF         DP
    Small (10)       0.019ms     0.019ms     0.2ms
    Medium (50)      0.052ms     0.048ms     706ms
    Large (200)      0.221ms     0.213ms     30,000ms

Engineering recommendation:
    - Up to 50 jobs and time allows: use DP for guaranteed optimal
    - 50 or more jobs or real-time decisions: use Greedy
    - EDF is not recommended for profit maximization

---

## Correctness Verification

DP correctness is verified by a brute-force gold standard test.
The brute-force checker tries all 2^10 = 1,024 combinations on the
small dataset and confirms DP finds the same optimal answer (7,700 euros).

All 8 unit tests pass.

---

## Data Parameters

All datasets use random.seed(42) for full reproducibility.
Parameters are grounded in realistic MTO manufacturing logic:

    Parameter        Range           Justification
    Duration         1 to 4 hours    Typical machining task duration
    Deadline         dur+1 to 8hrs   Same-shift delivery commitment
    Profit           500 to 5000     Realistic MTO order revenue
    Dependency rate  30 percent      Reflects real production chains
    Shift length     8 hours         Standard production shift

---

## Requirements

    Python 3.8 or above
    matplotlib
    numpy

Both matplotlib and numpy are included with Anaconda.
No additional installation required if using Anaconda.