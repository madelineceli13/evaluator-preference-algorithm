"""ICLR review-report analysis (ICLR 2025 and ICLR 2026), consolidated.

Runs the full analysis for both ICLR years:
  * loads per-review criteria + overall rating (and accept/reject decisions, when
    available),
  * fits the monotone "community preference function" (run_algorithm),
  * writes debiased reviews (per-review commensuration bias),
  * computes criterion importance (marginal-impact weights + Shapley values),
  * generates the report figures, and
  * (2026 only -- 2025 has no paper_id/decision data) computes every number quoted
    in the report and writes it to a text file.

Usage (from the repo root; needs cvxpy available):
    python iclr_analysis.py                # both years; reuses any saved fit
    python iclr_analysis.py --full          # force a refit for both years
    python iclr_analysis.py --reuse-fit     # skip the expensive Shapley refit

Everything is written under data/case_study/, with every filename tagged by
year (outputs_<year>/, debiased_reviews_<year>.csv, criteria_importance_<year>.csv,
plots/*_<year>.png, findings_<year>.txt) so 2025 and 2026 coexist in one directory.

Pipeline overview (see main() at the bottom): load -> fit_and_debias -> make_plots
-> compute_findings.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from itertools import product

# The three ICLR review criteria (each scored 1-4) and the overall recommendation.
# CRITERIA is the column order used for fitting; CRITERIA_ORDER is the display
# order used in the plots (contribution, then soundness, then presentation).
CRITERIA = ['soundness', 'presentation', 'contribution']
CRITERIA_ORDER = ['contribution', 'soundness', 'presentation']
OVERALL = 'rating'

OUTDIR = 'data/case_study'

# Everything that differs between the two years lives here; the rest of the code is
# year-agnostic and reads from this dict. Note the two years use different overall
# rating scales and different accept/reject decision labels. 2025 has no paper_id
# or decision column at all, so decision=None and compute_findings is skipped for it.
CONFIG = {
    2026: dict(
        csv='../data/ICLR/ICLR_reviews_2026.csv',
        decision='column',                        # decision already a column in the csv
        rating_support=[0, 2, 4, 6, 8, 10],
        accept={'ICLR 2026 Poster', 'ICLR 2026 Oral'},
        reject={'Submitted to ICLR 2026', 'ICLR 2026 Conference Withdrawn Submission'},
        headline_rate='27.4%',
    ),
    2025: dict(
        csv='../data/ICLR/ICLR_reviews_2025.csv',
        decision=None,                             # no paper_id/decision data available
    ),
}


def load(year):
    """Load the per-review table (one row per review).

    Rows missing any criterion or the overall rating are dropped. 'original row
    number' preserves the pre-drop index so a flagged review can be traced back to
    the source file.
    """
    cfg = CONFIG[year]
    df = pd.read_csv(cfg['csv'])
    df = df.dropna(subset=CRITERIA + [OVERALL], how='any').reset_index(drop=True)
    df['original row number'] = df.index
    return df, cfg


def fit_and_debias(year, df, cfg, reuse_fit):
    """Fit the community preference function, then write debiased reviews + importance.

    Steps:
      1. run_algorithm: cross-validate + fit the monotone criteria->rating function,
         saving it to outputs_<year>/cv_df.csv; empirical_simulation adds the
         train/test split used for the noise-floor / linearity numbers.
      2. iso_fit_many: apply that learned function to every review's criteria scores
         to get its "updated rating" (what the community norm would have given), and
         define commensuration bias = |actual rating - updated rating|.
      3. weight_vector + compute_shapley_values: two measures of criterion importance.
    --reuse-fit skips 1 and 3 (the expensive parts) when their outputs already exist.
    """
    from functions import iso_fit_many, empirical_simulation, run_algorithm
    from functions_interpretation import compute_shapley_values, weight_vector
    from scipy.stats import rankdata

    fit_dir = f'{OUTDIR}/outputs_{year}/'
    tt_dir = f'{OUTDIR}/outputs_train_test_{year}/'
    os.makedirs(fit_dir, exist_ok=True)
    X, Y = np.asarray(df[CRITERIA]), np.asarray(df[OVERALL])   # X = criteria, Y = overall rating
    cv_path = f'{fit_dir}cv_df.csv'

    # --- 1. fit the preference function (unless reusing a saved fit) ---
    if not (reuse_fit and os.path.exists(cv_path)):
        run_algorithm(X, Y, fit_dir, CRITERIA, OVERALL)
        os.makedirs(tt_dir, exist_ok=True)
        empirical_simulation(X, Y, tt_dir, f'ICLR {year}')

    # --- 2. debiased reviews: rating the community norm implies, and the gap to it ---
    cv_df = pd.read_csv(cv_path)                                # the learned function (unique criteria -> rating)
    corrected = iso_fit_many(X, np.asarray(cv_df[CRITERIA]), np.asarray(cv_df[OVERALL]))
    df['updated rating'] = np.round(corrected, 3)              # criteria-implied rating
    df['absolute difference'] = np.round(np.abs(Y - corrected), 3)   # commensuration bias
    df['bias percentile'] = np.round(rankdata(np.abs(Y - corrected), method='average') / len(df) * 100, 2)
    cols = ['original row number'] + [c for c in df.columns if c != 'original row number']

    # sorted most-biased first, so program chairs can scan the top of the file
    df.sort_values('absolute difference', ascending=False)[cols].to_csv(
        f'{OUTDIR}/debiased_reviews_{year}.csv', index=False)

    # --- 3. criterion importance (weights + Shapley); Shapley refit is the slow part ---
    imp_path = f'{OUTDIR}/criteria_importance_{year}.csv'
    if reuse_fit and os.path.exists(imp_path):
        print(f'reusing saved importance: {imp_path}')
    else:
        weights = weight_vector(cv_df)
        shap = compute_shapley_values(X, Y)   # slow: refits for every subset of criteria
        pd.DataFrame({'criterion': CRITERIA, 'shapley_value': shap, 'weight_vector': weights}
                     ).to_csv(imp_path, index=False)
    return cv_df


def make_plots(year):
    """Generate the report figures from the saved fit/debiased reviews: the
    commensuration-bias CDF, the marginal-effect heatmap, and the criterion
    importance bar chart. (Reuses the shared plotting code in interpretation_plots.py.)"""
    from plot_interpretation import plot_cdf, criterion_marginal_effect_heatmap, plot_criteria_importance
    os.makedirs(f'{OUTDIR}/plots', exist_ok=True)   # make the dir the plots actually go in
    d = pd.read_csv(f'{OUTDIR}/debiased_reviews_{year}.csv')
    plot_cdf(d['absolute difference'].values, OUTDIR, year)
    criterion_marginal_effect_heatmap(OUTDIR, year, np.asarray(d[CRITERIA_ORDER]), CRITERIA_ORDER, OVERALL)
    plot_criteria_importance(OUTDIR, year, CRITERIA_ORDER)


def grid_predictions(cv_df):
    """Predict the overall rating for every point on the full [1..4]^3 criteria grid.

    The fit (cv_df) only stores a value for criteria combinations that actually
    occurred; iso() fills in any unseen combination with the midpoint of the largest
    dominated value and the smallest dominating value (the same monotone
    interpolation rule used elsewhere). Returns {(s,p,c): predicted_rating}, which
    the gatekeeper / rising-bar / accept-recipe findings below all read from.
    """
    xv, fv = np.asarray(cv_df[CRITERIA]), np.asarray(cv_df[OVERALL])

    def iso(z):
        e = np.all(xv == z, axis=1)
        if e.any():                                    # exact match: use the fitted value
            return fv[e][0]
        A = fv[np.all(xv <= z, axis=1)]                # values at dominated points
        B = fv[np.all(xv >= z, axis=1)]                # values at dominating points
        return ((A.max() if len(A) else fv.min()) + (B.min() if len(B) else fv.max())) / 2
    return {x: iso(np.array(x)) for x in product(range(1, 5), repeat=3)}


def compute_findings(year, cfg, cv_df):
    """Compute every number quoted in the report, print them, and save to a text file.

    Each bracketed tag ([1]..[7]) corresponds to a section of the report. L is a list
    of output lines that is joined and written at the end. Requires decision data
    (2026 only -- see CONFIG); callers should skip this for years with decision=None.
    """
    d = pd.read_csv(f'{OUTDIR}/debiased_reviews_{year}.csv')
    imp = pd.read_csv(f'{OUTDIR}/criteria_importance_{year}.csv')
    L = [f'ICLR {year} findings  (n={len(d)} reviews, {d.paper_id.nunique()} papers)', '=' * 64]

    # [1] Which criteria matter most: normalize each importance measure to a share
    #     that sums to 1, and report the ranking (report claim: contribution > soundness > presentation).
    imp['shapley_share'] = imp.shapley_value / imp.shapley_value.sum()
    imp['weight_share'] = imp.weight_vector / imp.weight_vector.sum()
    L += ['[1] IMPORTANCE (share of total):',
          imp[['criterion', 'shapley_share', 'weight_share']].round(3).to_string(index=False),
          '    ranking: ' + ' > '.join(imp.sort_values('shapley_value', ascending=False).criterion)]

    # [2/3] Is the aggregation linear? Compare the CV-selected (best) regularization
    #       level against the fully-linear model (lambda = inf). "linear worst" means
    #       the linear model has the largest validation error of every level tried.
    cv = pd.read_csv(f'{OUTDIR}/outputs_train_test_{year}/CV_summary.csv')
    best, lin = cv.loc[cv.val_mse.idxmin()], cv[cv['lambda'] == np.inf].iloc[0]
    L += ['', '[2/3] LINEARITY: optimal lambda %.3g (val %.3f) vs linear %.3f (%.1f%% worse); linear worst=%s'
          % (best['lambda'], best.val_mse, lin.val_mse, 100 * (lin.val_mse - best.val_mse) / lin.val_mse,
             lin.val_mse == cv.val_mse.max())]

    # [4] How well is the mapping learned? Held-out MSE vs the irreducible floor
    #     (the lowest error any function of the criteria could achieve). R^2 is reported
    #     for reference only (the report no longer claims criteria "informativeness").
    st = pd.read_csv('results/empirical_stats_rebuttal.csv').iloc[0]
    var = d['rating'].var()
    L += ['', '[4] ESTIMATE QUALITY: held-out MSE %.3f vs irreducible floor %.3f (within %.1f%%); R^2 %.3f'
          % (st.MSE_cv, st.noise, 100 * (st.MSE_cv / st.noise - 1), 1 - st.MSE_cv / var)]

    # Everything below reads predictions on the full criteria grid.
    F = grid_predictions(cv_df)

    # [rising bar] Average effect of a +1 on one criterion, split by whether the OTHER
    #     two criteria are low (1-2) or high (3-4). The effect being larger in the "high"
    #     case is the "grades on a curve that steepens" finding.
    for lab, rng in [('others low (1-2)', [1, 2]), ('others high (3-4)', [3, 4])]:
        eff = []
        for i in range(3):                              # i = the criterion we bump by +1
            oth = [k for k in range(3) if k != i]       # the other two criteria
            for a in range(1, 4):                       # bump i from a -> a+1
                for o1 in rng:
                    for o2 in rng:                      # sweep the others over the low/high band
                        x = [0, 0, 0]; x[oth[0]] = o1; x[oth[1]] = o2
                        xa = list(x); xa[i] = a; xb = list(x); xb[i] = a + 1
                        eff.append(F[tuple(xb)] - F[tuple(xa)])
        L.append('[rising bar] %s: mean +1 effect %.2f' % (lab, np.mean(eff)))

    # [gatekeeper] Best achievable rating when one criterion is pinned at its floor (1).
    #     A low cap means that criterion "gates" the outcome. Ceiling = all criteria at 4.
    L.append('[gatekeeper] ceiling(4,4,4)=%.2f | ' % F[(4, 4, 4)]
             + ' '.join('%s=1 cap %.2f' % (c, max(v for x, v in F.items() if x[i] == 1))
                        for i, c in enumerate(CRITERIA)))

    # [accept recipe] Among all criteria combinations the function maps to >= 6, the
    #     minimum value each criterion takes (which criteria must be high to reach 6+).
    for T in [6]:
        hi = [x for x, v in F.items() if v >= T]
        if hi:
            L.append('[accept recipe] rating>=%d needs (min per criterion): %s'
                     % (T, dict(zip(CRITERIA, np.array(hi).min(0)))))

    # [6] Commensuration bias: how far reviews sit from their criteria-implied rating.
    #     Thresholds are in raw rating points; 'signed' splits over- vs under-rating.
    b = d['absolute difference']
    L += ['', '[6] COMMENSURATION BIAS: median %.2f mean %.2f | >=1pt %.1f%% >=2pt %.1f%% >=3pt %.1f%% >=4pt %.1f%%'
          % (b.median(), b.mean(), *[100 * (b >= t).mean() for t in [1, 2, 3, 4]])]
    signed = d['rating'] - d['updated rating']
    L.append('    over-rated >=2pt %.1f%% | under-rated <=-2pt %.1f%%'
             % (100 * (signed >= 2).mean(), 100 * (signed <= -2).mean()))

    # [5] Same scores, different ratings: take the most common criteria profile (the mode),
    #     show how widely its overall ratings still vary, and report the pooled
    #     within-profile standard deviation (a direct read of the review noise).
    g = d.groupby(CRITERIA)['rating']; cnt = g.count(); mode = cnt.idxmax()
    sub = d[(d[CRITERIA] == list(mode)).all(axis=1)]
    wv = (g.var(ddof=1) * (cnt - 1)).sum() / (cnt[cnt >= 2].sum() - len(cnt[cnt >= 2]))
    L += ['', '[5] SAME SCORES: mode profile %s n=%d, ratings %s; within-profile sd %.2f'
          % (tuple(int(x) for x in mode), len(sub), sub['rating'].value_counts().sort_index().to_dict(), np.sqrt(wv))]

    # [7] Decision-flip on real decisions. Model acceptance as the top submissions by
    #     mean rating, with the cutoff set to reproduce the real acceptance RATE among
    #     decided papers. Then ask: for what fraction of papers would a single reviewer
    #     moving their score by one rating level change the accept/reject outcome?
    support = sorted(cfg['rating_support'])
    # "one level" = the adjacent value on the (possibly uneven) rating scale
    up = {support[i]: support[min(i + 1, len(support) - 1)] for i in range(len(support))}
    dn = {support[i]: support[max(i - 1, 0)] for i in range(len(support))}
    # collapse reviews to one row per paper: its decision and the list of overall scores
    pap = d.groupby('paper_id').agg(dec=('decision', 'first'), scores=('rating', list)).reset_index()
    cl = pap[pap.dec.isin(cfg['accept'] | cfg['reject'])].copy()   # keep only decided papers
    cl['accept'] = cl.dec.isin(cfg['accept'])
    cl['mean'] = cl.scores.apply(np.mean); cl['n'] = cl.scores.apply(len)
    rate = cl.accept.mean()                                        # real accept rate among decided
    thr = cl['mean'].quantile(1 - rate)                            # mean-rating cutoff reproducing it
    n_acc = int(cl.accept.sum())
    # acceptance under the threshold model (top submissions by mean rating),
    # consistent with how the cutoff is defined; the real accept RATE sets thr.
    flips = 0
    for _, r in cl.iterrows():
        s = np.array(r['scores'], float); m, n = r['mean'], r['n']
        a = m >= thr                                              # accepted under the model?
        for i in range(n):                                        # try moving each reviewer one level toward the boundary
            new = up[s[i]] if not a else dn[s[i]]
            if ((m + (new - s[i]) / n) >= thr) != a:              # did that flip the accept/reject side?
                flips += 1; break
    step = np.median(np.diff(support))                            # a typical one-level step size
    band = (np.abs(cl['mean'] - thr) <= step / cl['n']).mean()    # papers within one reviewer-step of the cutoff
    L += ['', '[7] DECISION FLIP (real decisions): %d decided papers; accepted %d (%.1f%% among decided; '
          'headline %s); threshold mean %.2f' % (len(cl), n_acc, 100 * rate, cfg['headline_rate'], thr),
          '    flip if one reviewer moves one level: %.1f%% | within one level of threshold: %.1f%%'
          % (100 * flips / len(cl), 100 * band)]

    # [8] Commensuration-bias decision flips. Two acceptance decisions on the SAME
    #     decided papers, both using the [7] top-mean-score threshold model at the real
    #     accept rate:
    #       biased world   -- accept top papers by mean of the ACTUAL ratings y_i;
    #       unbiased world -- accept top papers by mean of the criteria-implied ratings
    #                         (updated rating = f-hat(x_i), i.e. y_i with zero commensuration bias).
    #     Each world sets its OWN cutoff as the (1-rate) quantile of its own mean-score
    #     distribution, so both accept the same fraction; we then count papers whose
    #     accept/reject side differs between the two worlds.
    pap2 = d.groupby('paper_id').agg(
        dec=('decision', 'first'),
        biased=('rating', list),
        unbiased=('updated rating', list),          # updated rating = f-hat(x) = zero-bias score
    ).reset_index()
    cl2 = pap2[pap2.dec.isin(cfg['accept'] | cfg['reject'])].copy()   # decided papers only (as in [7])
    rate2 = cl2.dec.isin(cfg['accept']).mean()                        # real accept rate among decided (== rate in [7])
    cl2['mean_biased']   = cl2.biased.apply(np.mean)
    cl2['mean_unbiased'] = cl2.unbiased.apply(np.mean)
    thr_b = cl2['mean_biased'].quantile(1 - rate2)                    # cutoff reproducing the rate, biased world
    thr_u = cl2['mean_unbiased'].quantile(1 - rate2)                  # cutoff reproducing the rate, unbiased world
    acc_b = cl2['mean_biased']   >= thr_b                             # accepted under actual scores (baseline)
    acc_u = cl2['mean_unbiased'] >= thr_u                             # accepted under zero-bias scores
    swapped = int((acc_b != acc_u).sum())                            # papers that change accept/reject side
    b2r = int((acc_b & ~acc_u).sum())                                # accepted only in the biased world (accept->reject)
    r2b = int((~acc_b & acc_u).sum())                                # accepted only in the unbiased world (reject->accept)
    L += ['', '[8] COMMENSURATION-BIAS DECISION FLIPS (biased vs unbiased world): %d decided papers; '
          'accepted %d biased / %d unbiased at %.1f%% rate; thresholds %.2f biased / %.2f unbiased'
          % (len(cl2), int(acc_b.sum()), int(acc_u.sum()), 100 * rate2, thr_b, thr_u),
          '    papers swapped: %d (%.1f%%) -- %d accept->reject, %d reject->accept'
          % (swapped, 100 * swapped / len(cl2), b2r, r2b)]

    # write + echo the findings
    text = '\n'.join(L)
    with open(f'{OUTDIR}/findings_{year}.txt', 'w') as fh:
        fh.write(text + '\n')
    print(text)
    print(f'\n(written to {OUTDIR}/findings_{year}.txt)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reuse-fit', action='store_true', help='reuse a saved fit if present')
    ap.add_argument('--full', action='store_true',
                     help='force a refit even if a saved fit already exists for that year')
    a = ap.parse_args()

    for year, cfg in CONFIG.items():
        print(f'\n=== ICLR {year} ===')
        cv_path = f'{OUTDIR}/outputs_{year}/cv_df.csv'
        if a.full or not os.path.exists(cv_path):
            df, cfg = load(year)                              # 1. load reviews (+ decisions, if any)
            cv_df = fit_and_debias(year, df, cfg, a.reuse_fit)  # 2. fit, debias, importance
        else:
            cv_df = pd.read_csv(cv_path)                       # reload the saved fit
        make_plots(year)                                       # 3. figures
        if cfg['decision'] is not None:
            compute_findings(year, cfg, cv_df)                 # 4. the report numbers
        else:
            print(f'(skipping findings for {year}: no decision/paper_id data available)')

if __name__ == '__main__':
    main()
