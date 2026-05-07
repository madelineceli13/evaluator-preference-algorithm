# Code for Learning What Evaluators Value

This repository contains the code accompanying the paper *Learning What Evaluators Value: A Reliable
Approach to Modeling Evaluator Preferences*.

## Overview

This codebase implements the algorithm and empirical results for our paper. 

## Requirements

```
numpy
pandas
scikit-learn
matplotlib
scipy
```

Install dependencies via:

```bash
pip install numpy pandas scikit-learn matplotlib scipy
```

## Data

The two primary datasets used in this paper are:

- **Tripadvisor (HotelRec):** Available at https://github.com/diegoantognini/HotelRec
- **ICLR peer reviews (human + LLM):** Available at https://huggingface.co/datasets/IntelLabs/AI-Peer-Review-Detection-Benchmark

Download these datasets and place them in a `data/` directory before running the experiments.

## Repository Structure

```
├── data/                   # Place downloaded datasets here
├── plots/                  # Output figures
├── results/                # Output results (CSVs)
├── empirical_stats.csv     # Precomputed prediction/reducible error statistics
├── noise_summary.csv       # Bootstrapped irreducible error estimates
├── residuals_human.csv     # Preference misalignment estimates
├── other                   # Code for all figures and results in the paper
└── README.md
```

## Reproducing Results

### Tripadvisor experiments
Runs the comparison of our algorithm vs. linear regression on the HotelRec dataset and saves results to `empirical_stats.csv`:

```bash
python empirical_hotelrec.py
```

### ICLR LLM experiments
Runs the consistency and preference alignment analysis on the ICLR peer review dataset:

```bash
python empirical_LLMs.py
```

### Synthetic simulations
Reproduces the synthetic preference simulations (linear, Cobb-Douglas, and Leontief utilities) from the appendix:

```bash
python synthetic_simulations.py
```

### Figures
Reproduces all figures in the paper: use plot_synthetic (for synthetic simulations), plot_understanding (for criteria plot in the appendix for the Tripadvisor data), plots (for other plots)

## Computational Requirements

All experiments were run locally on a personal computer. No GPUs or specialized infrastructure are required. All experiments complete within a few hours on standard consumer hardware.