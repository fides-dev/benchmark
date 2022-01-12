import fides.constants

from comparison import MODELS
from evaluate import (
    ANALYSIS_ALGOS, ALGO_PALETTES, CONVERGENCE_THRESHOLDS, get_stats_file
)

import h5py
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.scale import SymmetricalLogTransform
from itertools import groupby


def glen(grouper):
    return sum(1 for _ in grouper)


def max_streak(vector):
    run_len = [glen(run) for val, run in groupby(vector) if val]
    if not run_len:
        return 0
    return np.max(run_len)


FIDES_MU = fides.constants.DEFAULT_OPTIONS[fides.Options.MU]
FIDES_ETA = fides.constants.DEFAULT_OPTIONS[fides.Options.ETA]

STATS = {
    'accepted':
        lambda data: np.sum(data['accept'][:]),
    'no_hess_update':
        lambda data: np.logical_and.reduce((
            data['accept'][:],
            data['hess_update_min_ev'][:] == 0.0,
            data['hess_update_max_ev'][:] == 0.0,
        )).sum(),
    'no_hess_update_internal':
        lambda data: np.logical_and.reduce((
            data['accept'][:],
            data['hess_update_min_ev'][:] == 0.0,
            data['hess_update_max_ev'][:] == 0.0,
            data['reflections'][:] == 0
        )).sum(),
    'no_hess_update_border':
        lambda data: np.logical_and.reduce((
            data['accept'][:],
            data['hess_update_min_ev'][:] == 0.0,
            data['hess_update_max_ev'][:] == 0.0,
            data['reflections'][:] > 0
        )).sum(),
    'no_hess_struct_update':
        lambda data: np.logical_and.reduce((
            data['accept'][:],
            data['hess_struct_update_min_ev'][:] == 0.0,
            data['hess_struct_update_max_ev'][:] == 0.0,
        )).sum(),
    'no_tr_update_int_sol':
        lambda data: np.logical_and.reduce((
            data['tr_ratio'][:] > FIDES_MU,
            data['iterations_since_tr_update'][:] > 0,
        )).sum(),
    'no_tr_update_tr_ratio':
        lambda data: np.logical_and.reduce((
            data['tr_ratio'][:] < FIDES_ETA,
            data['tr_ratio'][:] > FIDES_MU,
            data['iterations_since_tr_update'][:] > 0,
        )).sum(),
    'streak_no_tr_update_tr_ratio':
        lambda data: max_streak(np.logical_and.reduce((
            data['tr_ratio'][:] < FIDES_ETA,
            data['tr_ratio'][:] > FIDES_MU,
            data['iterations_since_tr_update'][:] > 0,
        ))),
    'neg_ev':
        lambda data: np.sum(
            data['hess_min_ev'][:] < -np.spacing(1)*data['hess_max_ev']
        ),
    'singular_shess':
        lambda data: np.sum(data['cond_shess'][:] > 1 / np.spacing(1)),
    'posdef_newt':
        lambda data: np.sum(data['posdef'][:]),
    'degenerate_subspace':
        lambda data: np.logical_and.reduce((
            data['subspace_dim'][:] == 1,
            np.logical_not(data['newton'][:]),
            data['step_type'][:] == b'2d',
        )).sum() / np.sum(data['step_type'][:] == b'2d'),
    'newton_steps':
        lambda data: np.logical_and(
            data['newton'][:],
            data['step_type'][:] == b'2d',
        ).sum() / np.sum(data['step_type'][:] == b'2d'),
    'gradient_steps':
        lambda data: np.sum(
            data['step_type'][:] == b'g'
        ) / np.sum(np.logical_and(
            data['step_type'][:] != b'2d',
            data['step_type'][:] != b'nd',
        )),
    'border_steps':
        lambda data: np.sum(np.logical_and(
            data['step_type'][:] != b'2d',
            data['step_type'][:] != b'nd',
        )),
    'converged':
        lambda data, fmin:
            np.min(data['fval'][:]) < fmin + CONVERGENCE_THRESHOLDS[1],
    'integration_failure':
        lambda data: np.sum(np.logical_not(
            np.isfinite(data['fval'][:])
        )),
}

analysis_stats = {
    'curv': [
        'no_hess_update',
        'no_tr_update_tr_ratio', 'streak_no_tr_update_tr_ratio',
        'singular_shess', 'neg_ev',
        'newton_steps',
        'integration_failure'
    ],
    'hybrid': [
        'no_hess_update', 'no_hess_struct_update',
        'no_tr_update_tr_ratio',
        'singular_shess', 'neg_ev',
        'newton_steps', 'gradient_steps'
    ],
    'hybridB': [
        'no_hess_update',
        'no_tr_update_tr_ratio', 'streak_no_tr_update_tr_ratio'
    ],
    'stepback': [
        'no_tr_update_tr_ratio', 'no_tr_update_int_sol',
        'singular_shess',
        'gradient_steps', 'border_steps',
        'integration_failure'
    ],
}


def read_stats(model_name, optimizer, analysis):
    stats_file = get_stats_file(model_name, optimizer)
    print(f'loading {stats_file}')
    with h5py.File(stats_file, 'r') as f:
        fmin = np.min([
            np.min(data['fval'][:])
            for data in f.values()
        ])
        stats = pd.DataFrame([{
            **{'model': model_name,
               'optimizer': optimizer,
               'iter': data['fval'].size},
            **{stat:
               STATS[stat](data)/data['fval'].size + 1e-7
               if stat not in ['converged', 'degenerate_subspace',
                               'newton_steps', 'gradient_steps'] else
               STATS[stat](data)/data['fval'].size + 1e-7
               if stat != 'converged' else
               STATS[stat](data, fmin)
               for stat in analysis_stats[analysis] + ['converged']}
        } for data in f.values()])
    return stats


for analysis, algos in ANALYSIS_ALGOS.items():
    if analysis not in analysis_stats:
        continue
    stats = [
        read_stats(model, opt, analysis)
        for model in MODELS
        for opt in algos
        if opt.startswith('fides') and os.path.exists(
            get_stats_file(model, opt)
        )
    ]
    if not stats:
        continue
    all_stats = pd.concat(stats)
    df = pd.melt(all_stats, id_vars=['optimizer', 'model', 'iter',
                                     'converged'],
                 value_vars=analysis_stats[analysis])

    grid = sns.FacetGrid(
        row='model',
        col='variable',
        hue='optimizer',
        hue_order=algos,
        palette=ALGO_PALETTES[analysis],
        margin_titles=True,
        legend_out=True,
        despine=True,
        data=df
    )
    grid.map_dataframe(
        sns.kdeplot,
        x='iter',
        y='value',
        levels=5,
        thresh=0.2,
        alpha=0.5,
        log_scale=True,
        cut=0,
    )
    grid.map_dataframe(
        sns.scatterplot,
        x='iter',
        y='value',
        markers='X',
        edgecolors='none',
        alpha=0.5,
        size='converged',
        sizes={True: 12, False: 0},
    )
    grid.set(xscale='log', yscale='log', ylim=(1e-8, 10))
    grid.add_legend()
    plt.tight_layout()
    plt.savefig(os.path.join(
        'evaluation',
        f'stats_analysis_{analysis}.pdf'
    ))
    print(f'{analysis} done.')


