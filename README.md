# Code for Learning What Evaluators Value

Code for *Learning What Evaluators Value: A Reliable Approach to Modeling Evaluator Preferences*,
plus a follow-up ICLR peer-review interpretability case study.

## Requirements

Core: `numpy pandas scipy scikit-learn matplotlib cvxpy pyarrow`
Optional (only for the NN/GBM baselines in `additional_models.py`, `monotone_nn.py`,
`monotone_gbm.py`, `synthetic_simulations.py`): `torch xgboost mononet`

```bash
pip install -r requirements.txt   # core + optional
pip install "mononet[torch]"      # mononet needs this extras syntax
```

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
├── functions.py               # core library (regression, CV, bootstrap noise) -- everything imports this
├── empirical_*.py             # paper reproduction: hotelrec, LLM, NASA (+ evaluator-type), synthetic
├── additional_models.py, monotone_nn.py, monotone_gbm.py, synthetic_simulations.py
│                               # optional NN/GBM baselines
├── plot_synthetic.py, plot_understanding.py, plots.py   # paper figures
├── iclr_analysis.py           # ICLR case study entry point (see below)
├── interpretation_functions.py, interpretation_plots.py # its importance metrics + figures
├── data/                      # derived outputs (see Data section)
├── results/                   # output CSVs + process_data.py (prints result tables)
└── plots/                     # output figures
```

## Reproducing the Paper's Results

```bash
python empirical_hotelrec.py               # Tripadvisor
python empirical_LLM.py                    # ICLR LLM-vs-human reviews
python empirical_nasa_ICLR.py              # NASA solver/evaluator
python empirical_nasa_evaluator_type.py    # ... broken down by evaluator expertise
python empirical_synthetic.py              # synthetic simulations (appendix)

# optional NN/GBM baselines, needs torch/xgboost/mononet:
python monotone_nn.py
python monotone_gbm.py
python synthetic_simulations.py

python results/process_data.py             # print result tables from accumulated results/*.csv
```

Figures: `plot_synthetic.py`, `plot_understanding.py`, `plots.py`.

## ICLR Interpretability Case Study

Analyzes which review criteria (soundness, presentation, contribution) drive the overall ICLR
rating, and how far reviewers' ratings deviate from what the criteria alone imply
("commensuration bias"). Run from the repo root:

```bash
python iclr_analysis.py               # both ICLR 2025 and 2026; reuses any saved fit
python iclr_analysis.py --full        # force a refit for both years
python iclr_analysis.py --reuse-fit   # skip the expensive Shapley-value refit
```

Outputs go to `data/ICLR_case_study/`, filenames tagged by year (`outputs_<year>/`,
`debiased_reviews_<year>.csv`, `criteria_importance_<year>.csv`, `plots/*_<year>.png`,
`findings_<year>.txt`). 2025 has no `paper_id`/decision column, so its accept/reject
decision-flip findings are skipped -- it still gets the full set of plots.

## Computational Requirements

Runs locally, no GPU required. Most experiments finish within a few hours on consumer hardware;
the slowest single step is the ICLR case study's Shapley-value refit (~30s/year).
