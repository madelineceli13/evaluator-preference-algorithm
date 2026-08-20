import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import string
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import griddata


import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3d projection)
from matplotlib import cm
from matplotlib.colors import Normalize







plt.rcParams['font.size'] = 24
plt.rcParams['font.family'] = 'Times New Roman'

fontsize = 20 # for smaller components

styles = [
    {'color': 'red', 'marker': 'o', 'linestyle': '-'},
    {'color': 'orange', 'marker': 's', 'linestyle': '--'},
    {'color': 'green', 'marker': 'D', 'linestyle': ':'},
    {'color': 'blue', 'marker': '^', 'linestyle': '-.'},
]

# -------------------- ANALYSIS PART 1: COMMENSURATION BIAS PROBLEMS ------------------------------ # 

# Plot of commensuration bias distribution 

def plot_cdf(y, plot_dir_name, year):
    # Sort values to build the empirical CDF
    y = np.sort(y)
    n = len(y)
    cdf_values = np.arange(1, n + 1) / n  # i/n for sorted i-th value

    plt.figure(figsize=(8, 5))
    plt.plot(y, cdf_values, marker='.', linestyle='-')
    plt.xlabel("Review commensuration bias")
    plt.ylabel("CDF")
    plt.title("Empirical CDF of peer review commensuration bias")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()  # adjusts subplot params so labels/title fit within the figure
    plt.savefig(
        f'{plot_dir_name}/plots/commensuration_bias_{year}.png',
        bbox_inches='tight'  # trims whitespace and ensures nothing outside the axes gets clipped
    )


# -------------------- ANALYSIS PART 3: CRITERIA PLOTS ------------------------------ # 

def criterion_impact_plot(dir_name, X, CRITERIA, OVERALL):
    df_stats = pd.read_csv(f'{dir_name}/outputs_train_test/empirical_stats.csv')
    MSE = df_stats['MSE_cv'].values[0]
    SE = df_stats['SE_cv'].values[0]
    noise = df_stats['noise'].values[0]

    # obtain y_err metrics from results 
    y_err = error(MSE, SE, noise)

    print("y_err:", y_err)


    '''
    Show a plot that illustrates how the impact of changing one criterion score differs depending on the other criteria scores. 
    This is done by plotting the estimated overall score as a function of one criterion score, 
    and showing multiple curves for different fixed values of the other criteria scores. 
    '''
    x_min = int(X.min())
    x_max = int(X.max())

    # we make a big plot, with one subplot for each criterion, and multiple curves in each subplot for different fixed values of the other criteria
    num_criteria = len(CRITERIA)
    fig, axs = plt.subplots(1, num_criteria, figsize=(5*num_criteria, 5), sharey=True)

    df = pd.read_csv(f'{dir_name}/outputs/cv_df.csv')
    x_vals, f_vals = np.asarray(df.iloc[:, 1:-1]), np.asarray(df.iloc[:, -1])


    for i in range(num_criteria):
        for val in range(int(x_min), int(x_max)+1):
            mask = np.ones(len(x_vals), dtype=bool)
            for j in range(num_criteria):
                if j != i:
                    mask &= x_vals[:, j] == val 
            varying_vals = x_vals[mask, i].tolist()
            f_vals_CV = f_vals[mask].tolist()
            s = styles[val-1]
            print(f_vals_CV, varying_vals)
            axs[i].errorbar(varying_vals, f_vals_CV,
                        yerr = y_err,
                        color=s['color'],
                        marker=s['marker'],
                        linestyle=s['linestyle'],
                        linewidth=3,
                        markersize=10,
                        label=f'{val}/{x_max}')
        criterion_label = CRITERIA[i][0].upper() + CRITERIA[i][1:]
        axs[i].set_xlabel(f'{criterion_label} score', fontsize=fontsize)
        axs[i].grid(axis='y', alpha=0.3, linestyle='--')
        axs[i].tick_params(axis='both', labelsize=fontsize)  # tick numbers now match
    fig.subplots_adjust(right=0.9)   # subplots use left 82% of figure width
    handles, labels = axs[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels,
                         loc='center left',
                         bbox_to_anchor=(0.91, 0.5),
                         fontsize=fontsize,
                         title='Fixed value of other criteria')
    legend.get_title().set_fontsize(fontsize)

    axs[0].set_ylabel(f'Estimated overall {OVERALL}', fontsize=fontsize)
    fig.suptitle('Impact of increasing different criterion scores on the overall rating', fontsize=fontsize)
    plt.savefig(f'{dir_name}/plots/understanding_varying_criteria_impact.png', dpi=300, bbox_inches='tight')
    plt.close()

def criterion_marginal_effect_heatmap(dir_name, year, X, CRITERIA, OVERALL,
                                       fixed_values=(1, 2, 3, 4),
                                       cmap='Blues'):
    """
    Generate a heatmap showing the average marginal effect (mean slope,
    i.e. mean increment in estimated overall score per +1 step) of each
    criterion, as a function of the fixed value of the other criteria.
    Rows = criterion, columns = fixed value of other criteria (in CRITERIA
    order), cell value/color = average slope across the score range.
    """
    df = pd.read_csv(f'{dir_name}/outputs_{year}/cv_df.csv')
    x_vals = np.asarray(df[CRITERIA])   # select/order columns by name, not position
    f_vals = np.asarray(df[OVERALL])

    num_criteria = len(CRITERIA)
    x_max = int(np.ceil(X.max()))

    slope_matrix = np.full((num_criteria, len(fixed_values)), np.nan)

    for i in range(num_criteria):
        for col, val in enumerate(fixed_values):
            mask = np.ones(len(x_vals), dtype=bool)
            for j in range(num_criteria):
                if j != i:
                    mask &= x_vals[:, j] == val

            varying_vals = x_vals[mask, i]
            f_vals_CV = f_vals[mask]

            if len(varying_vals) < 2:
                continue

            # sort by criterion score so consecutive differences are
            # increments (score k -> k+1), not arbitrary row order
            order = np.argsort(varying_vals)
            sorted_scores = varying_vals[order]
            sorted_f = f_vals_CV[order]

            # average slope = mean of consecutive increments in f,
            # normalized by the step size in score (usually 1)
            score_diffs = np.diff(sorted_scores)
            f_diffs = np.diff(sorted_f)
            valid = score_diffs != 0
            if valid.sum() == 0:
                continue

            avg_slope = np.mean(f_diffs[valid] / score_diffs[valid])
            slope_matrix[i, col] = avg_slope

    fig, ax = plt.subplots(figsize=(1.6 * len(fixed_values) + 2, 1.2 * num_criteria + 1.5))

    im = ax.imshow(slope_matrix, cmap=cmap, aspect='auto')

    ax.set_xticks(np.arange(len(fixed_values)))
    ax.set_xticklabels([f'{val}/{x_max}' for val in fixed_values], fontsize=fontsize)
    ax.set_yticks(np.arange(num_criteria))
    ax.set_yticklabels([c[0].upper() + c[1:] for c in CRITERIA], fontsize=fontsize)

    ax.set_xlabel('Fixed value of other criteria', fontsize=fontsize)
    ax.set_title('Average marginal effect of each criterion\non the estimated overall rating',
                 fontsize=fontsize + 1)

    # annotate each cell with its numeric value, using contrasting text color
    vmin, vmax = np.nanmin(slope_matrix), np.nanmax(slope_matrix)
    mid = (vmin + vmax) / 2
    for i in range(num_criteria):
        for col in range(len(fixed_values)):
            v = slope_matrix[i, col]
            if np.isnan(v):
                continue
            text_color = 'white' if v > mid else 'black'
            ax.text(col, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=fontsize, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f'Avg. \u0394 {OVERALL} per +1 criterion score', fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    plt.tight_layout()
    plt.savefig(f'{dir_name}/plots/criterion_marginal_effect_heatmap_{year}.png',
                dpi=300, bbox_inches='tight')
    plt.close()


from matplotlib.colors import LinearSegmentedColormap, LightSource

def criteria_surface_plot_all_fixed(dir_name, X, CRITERIA, OVERALL,
                                     fixed_values=(1, 2, 3, 4),
                                     elev=25, azim=-60,
                                     edgecolor='grey', edge_lw=0.15, alpha=0.85):
    """
    Generate a figure with one 3D surface subplot per criterion, where each
    subplot fixes that criterion at each of `fixed_values` (overlaid as
    separate height-shaded surfaces) and varies the remaining two criteria
    on the x/y axes. Assumes len(CRITERIA) == 3.
    """
    assert len(CRITERIA) == 3, \
        "criteria_surface_plot_all_fixed assumes exactly 3 criteria"

    df = pd.read_csv(f'{dir_name}/outputs/cv_df.csv')
    x_vals = np.asarray(df.iloc[:, 1:-1])
    f_vals = np.asarray(df.iloc[:, -1])

    num_criteria = len(CRITERIA)
    panel_labels = list(string.ascii_lowercase)

    fig = plt.figure(figsize=(7 * num_criteria, 6))

    ls = LightSource(azdeg=315, altdeg=45)

    legend_handles = None
    fixed_max = int(np.ceil(X.max()))

    for idx, k_fixed in enumerate(range(num_criteria)):
        vary_idxs = [m for m in range(num_criteria) if m != k_fixed]
        i, j = vary_idxs[0], vary_idxs[1]

        ax = fig.add_subplot(1, num_criteria, idx + 1, projection='3d')

        x_min, x_max = int(np.floor(X[:, i].min())), int(np.ceil(X[:, i].max()))
        y_min, y_max = int(np.floor(X[:, j].min())), int(np.ceil(X[:, j].max()))

        panel_handles = []
        for val in fixed_values:
            mask = (x_vals[:, k_fixed] == val)

            xi = x_vals[mask, i]
            xj = x_vals[mask, j]
            z = f_vals[mask]

            s = styles[val - 1]
            base_color = s['color']
            # height-mapped colormap (white -> base hue) so topography
            # (peaks/valleys) is visible via color gradient + shading
            cmap_i = LinearSegmentedColormap.from_list(
                f'cmap_{val}', ['white', base_color]
            )

            ax.plot_trisurf(xi, xj, z, cmap=cmap_i,
                             edgecolor=edgecolor, linewidth=edge_lw,
                             alpha=alpha, antialiased=True,
                             shade=True, lightsource=ls)

            panel_handles.append(
                Patch(facecolor=base_color, edgecolor='none', label=f'{val}/{fixed_max}')
            )
        if legend_handles is None:
            legend_handles = panel_handles

        criterion_label_i = CRITERIA[i][0].upper() + CRITERIA[i][1:]
        criterion_label_j = CRITERIA[j][0].upper() + CRITERIA[j][1:]
        ax.set_xlabel(criterion_label_i, fontsize=fontsize, labelpad=10)
        ax.set_ylabel(criterion_label_j, fontsize=fontsize, labelpad=10)
        ax.set_zlabel(f'Overall {OVERALL}', fontsize=fontsize, labelpad=10)

        ax.set_xticks(np.arange(x_min, x_max + 1, 1))
        ax.set_yticks(np.arange(y_min, y_max + 1, 1))
        ax.view_init(elev=elev, azim=azim)

        fixed_label = CRITERIA[k_fixed][0].upper() + CRITERIA[k_fixed][1:]
        ax.set_title(f'Fixing {fixed_label}', fontsize=fontsize, pad=12)

        label = panel_labels[idx] if idx < len(panel_labels) else str(idx + 1)
        ax.text2D(0.5, -0.12, f'({label}) Varying '
                               f'{CRITERIA[i]} and {CRITERIA[j]}',
                  transform=ax.transAxes, ha='center', va='top', fontsize=fontsize)

    fig.subplots_adjust(right=0.88)
    legend = fig.legend(handles=legend_handles,
                         loc='center left',
                         bbox_to_anchor=(0.9, 0.5),
                         fontsize=fontsize,
                         title='Fixed value of\nother criterion')
    legend.get_title().set_fontsize(fontsize)

    fig.suptitle('Impact of varying pairs of criteria on the overall rating,\n'
                 'for different fixed values of the remaining criterion',
                 fontsize=fontsize + 1, y=1.05)

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.savefig(f'{dir_name}/plots/understanding_varying_criterion_pairs_by_fixed.png',
                dpi=300, bbox_inches='tight')
    plt.close()

def criteria_by_fixed_value_plot(dir_name, X, CRITERIA, OVERALL,
                                  fixed_values=(1, 2, 3, 4), ncols=2):
    """
    Generate a 2x2 grid of subplots, one per fixed value of the 'other'
    criteria, where each subplot overlays a line for each criterion
    (soundness/presentation/contribution) showing how the estimated overall
    score changes as that criterion is varied, holding the other criteria
    fixed at the subplot's value. Colors/styles distinguish criteria, drawn
    from the global `styles` list. Axes are shared: y-scale shared per row
    (left at matplotlib's default autoscale), x-scale shared per column.
    """
    df_stats = pd.read_csv(f'{dir_name}/outputs_train_test/empirical_stats.csv')
    MSE = df_stats['MSE_cv'].values[0]
    SE = df_stats['SE_cv'].values[0]
    noise = df_stats['noise'].values[0]
    y_err = error(MSE, SE, noise)

    df = pd.read_csv(f'{dir_name}/outputs/cv_df.csv')
    x_vals = np.asarray(df.iloc[:, 1:-1])
    f_vals = np.asarray(df.iloc[:, -1])

    num_criteria = len(CRITERIA)
    x_min, x_max = int(np.floor(X.min())), int(np.ceil(X.max()))

    n_plots = len(fixed_values)
    nrows = int(np.ceil(n_plots / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows),
                             sharex='col', sharey='row')
    axs = np.atleast_2d(axs)

    for idx, val in enumerate(fixed_values):
        row, col = divmod(idx, ncols)
        ax = axs[row, col]

        for i in range(num_criteria):
            mask = np.ones(len(x_vals), dtype=bool)
            for j in range(num_criteria):
                if j != i:
                    mask &= x_vals[:, j] == val
            varying_vals = x_vals[mask, i].tolist()
            f_vals_CV = f_vals[mask].tolist()

            s = styles[i]
            criterion_label = CRITERIA[i][0].upper() + CRITERIA[i][1:]
            ax.errorbar(varying_vals, f_vals_CV,
                        yerr=y_err,
                        color=s['color'],
                        marker=s['marker'],
                        linestyle=s['linestyle'],
                        linewidth=3,
                        markersize=10,
                        label=criterion_label)

        ax.set_title(f'Other criteria fixed at {val}/{x_max}', fontsize=fontsize)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.tick_params(axis='both', labelsize=fontsize)

        ax.set_xlim(x_min, x_max)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        if row == nrows - 1:
            ax.set_xlabel('Criterion score', fontsize=fontsize)
        if col == 0:
            ax.set_ylabel(f'Estimated overall {OVERALL}', fontsize=fontsize)

    for idx in range(n_plots, nrows * ncols):
        row, col = divmod(idx, ncols)
        axs[row, col].axis('off')

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.subplots_adjust(right=0.85)
    legend = fig.legend(handles, labels,
                         loc='center left',
                         bbox_to_anchor=(0.87, 0.5),
                         fontsize=fontsize,
                         title='Criterion')
    legend.get_title().set_fontsize(fontsize)

    fig.suptitle('Relative impact of varying different criterion', fontsize=fontsize + 1)

    plt.tight_layout(rect=[0, 0, 0.85, 0.96])
    plt.savefig(f'{dir_name}/plots/relative_impact_by_fixed_value.png',
                dpi=300, bbox_inches='tight')
    plt.close()


# -------------------- ANALYSIS PART 4: ADDITIONAL PLOTS (DRAFT MODE)------------------------------ # 


# criteria plots
def error(MSE, SE, noise):
    err = 1.96*SE/(2*np.sqrt(MSE - noise))
    return err





def criteria_surface_plot_by_row(dir_name, X, CRITERIA, OVERALL,
                                  grid_res=40, cmap='viridis', fontsize=14,
                                  elev=25, azim=-60, alpha=0.85):
    assert len(CRITERIA) == 3, "This function assumes exactly 3 criteria."

    df = pd.read_csv(f'{dir_name}/outputs/cv_df.csv')
    x_vals = np.asarray(df.iloc[:, 1:-1])
    f_vals = np.asarray(df.iloc[:, -1])

    pairs = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
    unique_vals = [np.unique(x_vals[:, c]) for c in range(3)]
    ncols = max(len(unique_vals[k]) for (_, _, k) in pairs)
    nrows = 3

    vmin, vmax = f_vals.min(), f_vals.max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)
    levels = np.arange(np.floor(vmin), np.ceil(vmax) + 1, 1)

    fig = plt.figure(figsize=(6 * ncols, 5.5 * nrows))
    axs = np.empty((nrows, ncols), dtype=object)
    panel_labels = list(string.ascii_lowercase)

    for row, (i, j, k) in enumerate(pairs):
        x_min, x_max = int(np.floor(X[:, i].min())), int(np.ceil(X[:, i].max()))
        y_min, y_max = int(np.floor(X[:, j].min())), int(np.ceil(X[:, j].max()))
        xi = np.linspace(x_min, x_max, grid_res)
        yi = np.linspace(y_min, y_max, grid_res)
        Xi, Yi = np.meshgrid(xi, yi)

        k_vals = unique_vals[k]
        k_max = int(X[:, k].max())

        for col in range(ncols):
            if col >= len(k_vals):
                continue

            ax = fig.add_subplot(nrows, ncols, row * ncols + col + 1, projection='3d')
            axs[row, col] = ax

            kval = k_vals[col]
            mask = x_vals[:, k] == kval
            xi_pts, xj_pts, z_pts = x_vals[mask, i], x_vals[mask, j], f_vals[mask]
            Zi = griddata((xi_pts, xj_pts), z_pts, (Xi, Yi), method='cubic')

            facecolors = cmap_obj(norm(Zi))

            # floor contour projection — flattened topo map sitting beneath the surface
            ax.contourf(Xi, Yi, Zi, levels=levels, cmap=cmap, norm=norm,
                        zdir='z', offset=vmin, alpha=0.6)

            # surface with a visible coarse wireframe so grid structure reads clearly
            ax.plot_surface(Xi, Yi, Zi, facecolors=facecolors,
                             edgecolor='black', linewidth=0.15, alpha=alpha,
                             shade=True, rstride=2, cstride=2, antialiased=True)

            label_i = CRITERIA[i][0].upper() + CRITERIA[i][1:]
            label_j = CRITERIA[j][0].upper() + CRITERIA[j][1:]
            label_k = CRITERIA[k][0].upper() + CRITERIA[k][1:]

            ax.set_xlabel(label_i, fontsize=fontsize, labelpad=10)
            ax.set_ylabel(label_j, fontsize=fontsize, labelpad=10)
            ax.set_zlabel(f'Estimated overall {OVERALL}', fontsize=fontsize - 2, labelpad=6)
            ax.set_xticks(np.arange(x_min, x_max + 1, 1))
            ax.set_yticks(np.arange(y_min, y_max + 1, 1))
            ax.set_zlim(vmin, vmax)  # keep floor contour flush with the axis bottom
            ax.view_init(elev=elev, azim=azim)
            ax.tick_params(axis='both', labelsize=fontsize - 3)
            ax.set_title(f"{label_k}={int(kval)}/{k_max}", fontsize=fontsize, pad=0)

    fig.subplots_adjust(right=0.88, hspace=0.35, top=0.92)
    fig.canvas.draw()

    for row, (i, j, k) in enumerate(pairs):
        n_active = len(unique_vals[k])
        row_axes = [axs[row, c] for c in range(n_active)]
        positions = [a.get_position() for a in row_axes]
        x_center = (positions[0].x0 + positions[-1].x1) / 2
        y_top = max(p.y1 for p in positions)

        label = panel_labels[row]
        label_i = CRITERIA[i][0].upper() + CRITERIA[i][1:]
        label_j = CRITERIA[j][0].upper() + CRITERIA[j][1:]
        fig.text(x_center, y_top + 0.005,
                  f'({label}) Varying {label_i} and {label_j}',
                  fontsize=fontsize + 3, ha='center', va='bottom', fontweight='bold')

    fig.suptitle('Impact of jointly varying pairs of criteria\n'
                  'across levels of the remaining criterion',
                  fontsize=fontsize + 6, y=1.0)

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(f'Estimated overall {OVERALL}', fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    plt.savefig(f'{dir_name}/plots/criteria_surface_plot_by_row.png',
                dpi=300, bbox_inches='tight')
    plt.close()

from scipy.interpolate import griddata
from matplotlib import cm

def criteria_contour_plot(dir_name, X, CRITERIA, OVERALL, pairs = None,
                           ncols=3, grid_res=100, n_levels=12,
                           cmap='viridis'):
    """
    Generate 2D filled contour plots showing how the estimated overall score
    varies when jointly varying pairs of criteria, with all other criteria
    fixed at their mode. Companion to criteria_surface_plot, but using flat
    2D contourf panels (with a shared colorbar) instead of 3D surfaces.
    """
    df = pd.read_csv(f'{dir_name}/outputs/cv_df.csv')
    x_vals = np.asarray(df.iloc[:, 1:-1])
    f_vals = np.asarray(df.iloc[:, -1])

    num_criteria = len(CRITERIA)
    criterion_modes = stats.mode(X, axis=0).mode

    if pairs is None:
        pairs = []
        for i in range(num_criteria-1):
            for j in range(i+1, num_criteria):
                pairs.append((i, j))

    n_plots = len(pairs)
    
    if ncols is None:
        ncols = np.ceil(np.sqrt(n_plots)).astype(int)
    nrows = int(np.ceil(n_plots / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axs = np.atleast_2d(axs)
    panel_labels = list(string.ascii_lowercase)

    # shared color scale across all panels so panels are directly comparable
    vmin, vmax = f_vals.min(), f_vals.max()

    # make levels at integer values
    levels = np.arange(np.floor(vmin), np.ceil(vmax) + 1, 1)

    last_cf = None

    for idx, (i, j) in enumerate(pairs):
        row, col = divmod(idx, ncols)
        ax = axs[row, col]

        mask = np.ones(len(x_vals), dtype=bool)
        for k in range(num_criteria):
            if k != i and k != j:
                mask &= x_vals[:, k] == criterion_modes[k]

        xi_pts = x_vals[mask, i]
        xj_pts = x_vals[mask, j]
        z_pts = f_vals[mask]

        x_min, x_max = int(np.floor(X[:, i].min())), int(np.ceil(X[:, i].max()))
        y_min, y_max = int(np.floor(X[:, j].min())), int(np.ceil(X[:, j].max()))

        xi = np.linspace(x_min, x_max, grid_res)
        yi = np.linspace(y_min, y_max, grid_res)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = griddata((xi_pts, xj_pts), z_pts, (Xi, Yi), method='cubic')

        cf = ax.contourf(Xi, Yi, Zi, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
        cs = ax.contour(Xi, Yi, Zi, levels=levels, colors='white',
                         linewidths=0.5, alpha=0.5)
        
        last_cf = cf

        criterion_label_i = CRITERIA[i][0].upper() + CRITERIA[i][1:]
        criterion_label_j = CRITERIA[j][0].upper() + CRITERIA[j][1:]
        ax.set_xlabel(criterion_label_i, fontsize=fontsize)
        ax.set_ylabel(criterion_label_j, fontsize=fontsize)
        ax.set_xticks(np.arange(x_min, x_max + 1, 1))
        ax.set_yticks(np.arange(y_min, y_max + 1, 1))
        ax.tick_params(axis='both', labelsize=fontsize)

        label = panel_labels[idx] if idx < len(panel_labels) else str(idx + 1)
        ax.set_title(f'({label}) Varying {CRITERIA[i]} and {CRITERIA[j]}',
                     fontsize=fontsize, y = 1.05)

    modes_str = ", ".join(
        f"{CRITERIA[k]}={int(criterion_modes[k])}/{x_max}" for k in range(num_criteria)
    )
    fig.suptitle('Indifference curves for different criterion pairs \n'
                 f'(other criterion fixed at its mode: {modes_str})',
                 fontsize=fontsize + 5, y=1.15)

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(last_cf, cax=cbar_ax)
    cbar.set_label(f'Estimated overall {OVERALL}', fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    plt.savefig(f'{dir_name}/plots/criteria_contour_plot.png',
                dpi=300, bbox_inches='tight')
    plt.close()



def criteria_contour_plot_by_row(dir_name, X, CRITERIA, OVERALL,
                                  grid_res=100, cmap='viridis', fontsize=14):
    """
    Assumes exactly 3 criteria. Each row corresponds to one of the 3 possible
    pairs; each column within a row corresponds to one value of the
    remaining (third) criterion.
    """
    assert len(CRITERIA) == 3, "This function assumes exactly 3 criteria."

    df = pd.read_csv(f'{dir_name}/outputs/cv_df.csv')
    x_vals = np.asarray(df.iloc[:, 1:-1])
    f_vals = np.asarray(df.iloc[:, -1])

    # (i, j, k): pair (i, j) plotted, k is the remaining criterion that varies by column
    pairs = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]

    unique_vals = [np.unique(x_vals[:, c]) for c in range(3)]
    ncols = max(len(unique_vals[k]) for (_, _, k) in pairs)
    nrows = 3

    vmin, vmax = f_vals.min(), f_vals.max()
    levels = np.arange(np.floor(vmin), np.ceil(vmax) + 1, 1)

    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), sharey='row')
    axs = np.atleast_2d(axs)
    panel_labels = list(string.ascii_lowercase)
    last_cf = None

    for row, (i, j, k) in enumerate(pairs):
        x_min, x_max = int(np.floor(X[:, i].min())), int(np.ceil(X[:, i].max()))
        y_min, y_max = int(np.floor(X[:, j].min())), int(np.ceil(X[:, j].max()))
        xi = np.linspace(x_min, x_max, grid_res)
        yi = np.linspace(y_min, y_max, grid_res)
        Xi, Yi = np.meshgrid(xi, yi)

        k_vals = unique_vals[k]
        k_max = int(X[:, k].max())

        for col in range(ncols):
            ax = axs[row, col]
            if col >= len(k_vals):
                ax.axis('off')
                continue

            kval = k_vals[col]
            mask = x_vals[:, k] == kval

            xi_pts, xj_pts, z_pts = x_vals[mask, i], x_vals[mask, j], f_vals[mask]
            Zi = griddata((xi_pts, xj_pts), z_pts, (Xi, Yi), method='cubic')

            cf = ax.contourf(Xi, Yi, Zi, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.contour(Xi, Yi, Zi, levels=levels, colors='white', linewidths=0.5, alpha=0.5)
            last_cf = cf

            label_i = CRITERIA[i][0].upper() + CRITERIA[i][1:]
            label_j = CRITERIA[j][0].upper() + CRITERIA[j][1:]
            label_k = CRITERIA[k][0].upper() + CRITERIA[k][1:]

            ax.set_xlabel(label_i, fontsize=fontsize)
            if col == 0:
                ax.set_ylabel(label_j, fontsize=fontsize)  # shared within the row
            ax.set_xticks(np.arange(x_min, x_max + 1, 1))
            ax.set_yticks(np.arange(y_min, y_max + 1, 1))
            ax.tick_params(axis='both', labelsize=fontsize)
            ax.set_title(f"{label_k}={int(kval)}/{k_max}", fontsize=fontsize, pad=10)

    # generous, fixed vertical spacing between rows so row-titles have room
    fig.subplots_adjust(right=0.88, hspace=0.45, top=0.92)
    fig.canvas.draw()  # need real axes positions before placing row titles

    for row, (i, j, k) in enumerate(pairs):
        n_active = len(unique_vals[k])
        left_pos = axs[row, 0].get_position()
        right_pos = axs[row, n_active - 1].get_position()
        x_center = (left_pos.x0 + right_pos.x1) / 2
        y_top = left_pos.y1

        label = panel_labels[row]
        label_i = CRITERIA[i][0].upper() + CRITERIA[i][1:]
        label_j = CRITERIA[j][0].upper() + CRITERIA[j][1:]
        fig.text(x_center, y_top + 0.025,
                  f'({label}) Varying {label_i} and {label_j}',
                  fontsize=fontsize + 3, ha='center', va='bottom', fontweight='bold')

    fig.suptitle('Indifference curves for Peer Review Preferences',
                  fontsize=fontsize + 6, y=1.0)

    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(last_cf, cax=cbar_ax)
    cbar.set_label(f'Estimated overall {OVERALL}', fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    plt.savefig(f'{dir_name}/plots/criteria_contour_plot_by_row.png',
                dpi=300, bbox_inches='tight')
    plt.close()




# ── Shared constants ─────────────────────────────────────────────────────────
BAR_WIDTH  = 0.6
COL_WIDTH  = 2.5
FIG_HEIGHT = 6

BAR_KWARGS = dict(
    width     = BAR_WIDTH,
    edgecolor = 'grey',
    linewidth = 1.2,
)

# One (colour, hatch) pair per criterion, cycled if there are more criteria
CRITERION_STYLES = [
    ('honeydew', '/'),
    ('lightblue',    '.'),
    ('thistle',      '|'),
]


def plot_criteria_importance(dir_name, year, order=None):
    df = pd.read_csv(f'{dir_name}/criteria_importance_{year}.csv')
    if order is not None:
        df = df.set_index('criterion').loc[order].reset_index()
    x = np.arange(len(df))
    n = len(df)

    styles  = [CRITERION_STYLES[k % len(CRITERION_STYLES)] for k in range(n)]
    colors  = [c for c, _ in styles]
    hatches = [h for _, h in styles]

    fig, ax = plt.subplots(figsize=(COL_WIDTH * n, FIG_HEIGHT))

    bars = ax.bar(x, df['shapley_value'], color=colors, **BAR_KWARGS)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    ax.set_ylabel('Shapley value')
    ax.set_xticks(x)
    ax.set_xticklabels(df['criterion'], rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.set_xlim(-0.5, n - 0.5)
    ax.axhline(0, color='grey', linewidth=1.2)

    fig.suptitle('Overall Criterion Importance')
    plt.tight_layout()
    plt.savefig(f'{dir_name}/plots/criteria_importance_{year}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


