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
from itertools import groupby


def glen(grouper):
    return sum(1 for _ in grouper)


def max_streak(vector):
    run_len = [glen(run) for val, run in groupby(vector) if val]
    if not run_len:
        return 0
    return np.max(run_len)


FIDES_MU = fides.constants.DEFAULT_OPTIONS.MU
FIDES_ETA = fides.constants.DEFAULT_OPTIONS.ETA

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
            data['tr_ratio'][:] > fides.constants.DEFAULT_OPTIONS,
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
        )).sum(),
    'newton_steps':
        lambda data: np.logical_and(
            data['newton'][:],
            data['step_type'][:] == b'2d',
        ).sum(),
    'gradient_steps':
        lambda data: np.sum(data['step_type'][:] == b'g'),
    'border_steps':
        lambda data: np.sum(np.logical_and(
            data['step_type'][:] != b'2d',
            data['step_type'][:] != b'nd',
        )),
    'converged':
        lambda data, fmin:
            np.min(data['fval'][:]) < fmin + CONVERGENCE_THRESHOLDS[1],
    'integration_failure':
        lambda data: np.sum(np.logical_not(np.isfinite(data['fval'][:]))),
}

analysis_stats = {
    'curv': [
        'no_hess_update',
        'no_update_tr_ratio', 'streak_no_tr_update_tr_ratio',
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
        'no_update_tr_ratio', 'streak_no_tr_update_tr_ratio'
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
            **{stat: STATS[stat](data) if stat != 'converged' else
               STATS[stat](data, fmin)
               for stat in analysis_stats[analysis]}
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
        data=df,
        row='model',
        col='variable',
        hue='optimizer',
        hue_order=algos,
        palette=ALGO_PALETTES[analysis],
        margin_titles=True,
        legend_out=True,
        despine=True,
    )
    grid.map_dataframe(
        sns.scatterplot,
        x='iter',
        y='value',
        style='converged',
        markers={True: 's', False: 'X'},
        alpha=0.2,
        s=8,
    )
    grid.set(xscale='log', yscale='symlog', ylim=(0, 1e5))
    grid.add_legend()
    plt.tight_layout()
    plt.savefig(os.path.join(
        'evaluation',
        f'stats_analysis_{analysis}.pdf'
    ))
    print(f'{analysis} done.')


