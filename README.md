# Code for Learning What Evaluators Value

Code for *Learning What Evaluators Value: A Reliable Approach to Modeling Evaluator Preferences*

## Requirements

Core: `numpy pandas scipy scikit-learn matplotlib cvxpy pyarrow`
Optional (only for the NN/GBM baselines in `additional_models.py`, `monotone_nn.py`,
`monotone_gbm.py`, `empirical_synthetic_nn_gbm.py`): `torch xgboost mononet`

## Data

Raw source data lives in a `data/` directory **one level above this repo** (sibling, not tracked
in git) -- scripts use paths like `'../data/ICLR/...'` and must be run from the repo root:

- **HotelRec (Tripadvisor):** `../data/hotelrec/*.parquet` -- https://github.com/diegoantognini/HotelRec
- **ICLR reviews:** `../data/ICLR/ICLR_reviews_<year>.csv`, `../data/ICLR/solver_evaluator_pairs_13.csv`,
  `../data/LLM reviews/*.csv` -- https://huggingface.co/datasets/IntelLabs/AI-Peer-Review-Detection-Benchmark

The `data/` directory *inside* this repo is different: it holds this codebase's own derived
outputs (fits, debiased reviews, plots) and is checked into git.

## Repository Structure

```
├── functions.py                              # core library (regression, CV, bootstrap noise) -- everything imports this
├── empirical_hotelrec.py, empirical_LLM.py, empirical_ICLR.py,
│   empirical_nasa.py, empirical_synthetic.py # paper reproduction scripts
├── additional_models.py, monotone_nn.py, monotone_gbm.py,
│   empirical_synthetic_nn_gbm.py             # optional NN/GBM baselines
├── plot_synthetic.py, plot_understanding.py,
│   plot_tripadvisor.py, tables.py            # paper figures + result tables
├── empirical_case_study.py                   # ICLR case study entry point (see below)
├── functions_interpretation.py, plot_interpretation.py # its importance metrics + figures
├── data/                                     # derived outputs (see Data section)
├── results/                                  # output CSVs
└── plots/                                    # output figures
```

## Reproducing the Paper's Results

```bash
python empirical_hotelrec.py               # Tripadvisor
python empirical_LLM.py                    # ICLR LLM-vs-human reviews
python empirical_ICLR.py                   # ICLR 2023/2024/2025 solver/evaluator
python empirical_nasa.py                   # NASA solver/evaluator (+ breakdown by evaluator expertise)
python empirical_synthetic.py              # synthetic simulations 

# optional NN/GBM baselines, needs torch/xgboost/mononet:
python monotone_nn.py
python monotone_gbm.py
python empirical_synthetic_nn_gbm.py
python tables.py                           # print result tables from accumulated results/*.csv
```

Figures: `plot_synthetic.py`, `plot_understanding.py`, `plot_tripadvisor.py`.

## ICLR Interpretability Case Study

Analyzes which review criteria (soundness, presentation, contribution) drive the overall ICLR
rating, and how far reviewers' ratings deviate from what the criteria alone imply
("commensuration bias"). Run from the repo root:

```bash
python empirical_case_study.py                # both ICLR 2025 and 2026; reuses any saved fit
python empirical_case_study.py --full         # force a refit for both years
python empirical_case_study.py --reuse-fit    # skip the expensive Shapley refit
```

Outputs go to `data/case_study/`, filenames tagged by year (`outputs_<year>/`,
`debiased_reviews_<year>.csv`, `criteria_importance_<year>.csv`, `plots/*_<year>.png`,
`findings_<year>.txt`). 2025 has no `paper_id`/decision column, so its accept/reject
decision-flip findings are skipped -- it still gets the full set of plots.

## Computational Requirements

Runs locally, no GPU required.