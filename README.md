# PARCO – Distributed Sparse Matrix–Vector Multiplication (SpMV)
- Author: Alessandro Gremes (ID 242330)
- Course: Parallel Computing 2025/2026
- University: University of Trento

![Language](https://img.shields.io/badge/language-C-blue.svg)
![Parallel](https://img.shields.io/badge/parallel-MPI-green.svg)
![MPI](https://img.shields.io/badge/MPI-MPICH_3.2.1-brightgreen.svg)
![Scheduler](https://img.shields.io/badge/scheduler-PBS-orange.svg)
![Scaling](https://img.shields.io/badge/scaling-strong%20%7C%20weak-blueviolet.svg)
![HPC](https://img.shields.io/badge/HPC-UniTN-red.svg)
![Reproducibility](https://img.shields.io/badge/reproducible-yes-success.svg)


## Project Overview

This project implements a **distributed Sparse Matrix–Vector Multiplication (SpMV)**
using **MPI** on a distributed-memory HPC system.

The matrix is distributed across processes using a **1D modulo-cyclic row distribution**,
while each process stores its local submatrix in **CSR (Compressed Sparse Row)** format.
The SpMV operation requires communication of remote vector elements, which is explicitly
managed through MPI collective communication.

The main objectives of the project are:
- to evaluate **strong and weak scaling behavior**,
- to study the impact of **load balancing** on irregular sparse matrices,
- to quantify **communication overhead** in distributed SpMV.
## Repository Structure

The repository is organized as follows:
```text
.
├── src/            # C source code (MPI-based SpMV implementation)
├── scripts/        # PBS job scripts and configuration files
├── results/        # Experimental results (CSV files)
├── plots/          # Generated plots (speedup, efficiency, scaling)
├── mtx/            # Sparse matrices (SuiteSparse / Matrix Market)
├── README.md
└── .gitignore
```
## Tasks and Objectives

As required by the deliverable, this project:

- implements a distributed SpMV using MPI
- uses an explicit data distribution strategy (1D modulo-cyclic)
- evaluates both strong and weak scaling
- performs multiple repetitions per configuration (≥10)
- reports maximum execution time across ranks
- ensures full reproducibility via scripts and fixed seeds

## Data Distribution Strategy

Rows are distributed across MPI ranks using a 1D modulo-cyclic scheme.
This strategy mitigates load imbalance on irregular sparse matrices by
interleaving rows with different sparsity patterns across processes,
at the cost of increased communication for remote vector elements.

## Implementation Details

- Local matrices are stored in CSR format.
- Each MPI rank owns only its local rows.
- Remote vector entries are exchanged at each iteration via MPI collectives.
- A warm-up iteration is executed before timing.
- Timings are measured using MPI_Wtime and reduced using MPI_MAX.


## Compilation

The code is written in C and uses **MPI** for parallelism.
Compilation is performed directly on the UniTN HPC cluster using `mpicc`.
This section describes the complete workflow to reproduce the experiments
from scratch on the UniTN HPC cluster.

In weak scaling experiments, the number of rows per MPI rank is kept constant,
while the total problem size grows proportionally to the number of processes.


## 🚀 **Execution and Reproducibility**

The experiments are fully reproducible:

- All parameters are controlled via environment variables
- Weak scaling uses a fixed random seed
- Multiple repetitions are executed for each configuration
- Results are stored in timestamped CSV files
- No manual post-processing is required during execution
  

The complete workflow, from matrix download to job submission, is documented in this README.
```bash
git clone git@github.com:alegrem123/PARCO2-Computing-2026-242330.git
cd PARCO2-Computing-2026-242330
module purge
module load gcc91
module load mpich-3.2.1--gcc-9.1.0
mkdir mtx
cd mtx
wget https://suitesparse-collection-website.herokuapp.com/MM/FEMLAB/poisson3Db.tar.gz
gzip -d poisson3Db.tar.gz
tar -xf poisson3Db.tar
wget https://suitesparse-collection-website.herokuapp.com/MM/Schmid/thermal2.tar.gz
gzip -d thermal2.tar.gz
tar -xf thermal2.tar
wget https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/kron_g500-logn21.tar.gz
gzip -d kron_g500-logn21.tar.gz
tar -xf kron_g500-logn21.tar
wget https://suitesparse-collection-website.herokuapp.com/MM/Williams/webbase-1M.tar.gz
gzip -d webbase-1M.tar.gz
tar -xf webbase-1M.tar
wget https://suitesparse-collection-website.herokuapp.com/MM/LAW/eu-2005.tar.gz
gzip -d eu-2005.tar.gz
tar -xf eu-2005.tar
cd ..
qsub scripts/run_spmv.pbs
qsub -v CONF=scripts/conf_weak.env scripts/run_spmv.pbs
cd results
ls

```
The provided PBS script is designed with sensible default parameters to allow immediate execution without requiring manual edits.
However, for strong scaling experiments, it is often necessary to run multiple configurations using different matrices, repetition counts, or process lists.


To support this, the script allows overriding its default parameters at submission time using environment variables passed via qsub -v as follows: 
```bash
qsub -v MODE=strong,MATRIX=mtx/poisson3Db/poisson3Db.mtx,RUNS=10,L=5,PLIST=1,2,4,8,16,32,64,128, scripts/run_spmv.pbs
```
The results are automatically stored in a timestamped CSV file under:
```text
results/strongScaling/<matrix_name>/strong_<date>.csv
```

## Input Data (Matrices)

The strong-scaling experiments use real-world sparse matrices from the **SuiteSparse Matrix Collection**.
Matrices are **not versioned** in the repository and must be downloaded manually, as specified below.

The following matrices are used for strong scaling:

- `poisson3Db` (Boeing): structured 3D stencil, memory-bound
- `FEM_3D_thermal2` (FEMLAB): finite-element matrix, irregular sparsity
- `kron_g500-logn21` (DIMACS10): synthetic Kronecker graph
- `webbase-1M` (LAW): large-scale web graph
- `eu-2005` (LAW): real-world web graph with irregular sparsity, suitable for scalable distributed SpMV.


Downloaded matrices are stored under `mtx/` and ignored by Git via `.gitignore`.

## Experimental Setup

All experiments were executed on the UniTN HPC cluster using distributed-memory nodes.

- Scheduler: PBS
- Queue: short_cpuQ
- Nodes: up to 32
- Cores per node: 4
- MPI ranks: up to 128
- MPI implementation: MPICH 3.2.1
- Compiler: GCC 9.1
- Compilation flags: -O0 -g (no optimizations)
- Timing routine: MPI_Wtime
- Metric: maximum execution time across ranks
## Output Format

Each experiment produces a CSV file with the following columns:

- `timestamp`: job execution timestamp
- `mode`: strong or weak scaling
- `case`: matrix name or weak-scaling configuration
- `np`: number of MPI processes
- `run`: repetition index
- `time_ms`: SpMV execution time in milliseconds
- `status`: execution status (OK, MPI_FAIL, PARSE_FAIL)

The reported execution time corresponds to the **maximum time across all MPI ranks**.

## Post-processing and Plots

Experimental results are post-processed to compute:
- average execution time
- 90th percentile execution time
- strong-scaling speedup
- weak-scaling efficiency

Plots are generated offline and stored under the `plots/` directory.

## Limitations

- No overlap of communication and computation is implemented.
- The focus is on correctness and scalability, not on low-level MPI tuning.
- Hybrid MPI+OpenMP is disabled (OMP_NUM_THREADS=1).




