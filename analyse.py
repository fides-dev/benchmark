from comparison import MODELS
from evaluate import (
    ANALYSIS_ALGOS, N_STARTS_FORWARD, ALGO_PALETTES, CONVERGENCE_THRESHOLD
)

import h5py
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import groupby


def get_stats_file(model_name, optimizer):
    return os.path.join(
        'stats',
        f'{model_name}__{optimizer}__{N_STARTS_FORWARD[0]}__STATS.hdf5'
    )


def glen(grouper):
    return sum(1 for _ in grouper)


def max_streak(vector):
    run_len = [glen(run) for val, run in groupby(vector) if val]
    if not run_len:
        return 0
    return np.max(run_len)


def read_stats(model_name, optimizer):
    stats_file = get_stats_file(model_name, optimizer)
    print(f'loading {stats_file}')
    with h5py.File(stats_file, 'r') as f:
        fmin = np.min([
            np.min(data['fval'][:])
            for data in f.values()
        ])
        stats = pd.DataFrame([{
            'model': model_name,
            'optimizer': optimizer,
            'iter': data['fval'].size,
            'frac_accepted': np.sum(data['accept'][:]) / data['fval'].size,
            'frac_no_hess_update': np.logical_and.reduce((
                data['accept'][:],
                data['hess_update_min_ev'][:] == 0.0,
                data['hess_update_max_ev'][:] == 0.0,
            )).sum() / np.sum(data['accept'][:]),
            'frac_no_hess_update_internal': np.logical_and.reduce((
                data['accept'][:],
                data['hess_update_min_ev'][:] == 0.0,
                data['hess_update_max_ev'][:] == 0.0,
                data['reflections'][:] == 0
            )).sum() / np.sum(data['accept'][:]),
            'frac_no_hess_update_border': np.logical_and.reduce((
                data['accept'][:],
                data['hess_update_min_ev'][:] == 0.0,
                data['hess_update_max_ev'][:] == 0.0,
                data['reflections'][:] > 0
            )).sum() / np.sum(data['accept'][:]),
            'frac_no_hess_struct_update': np.logical_and.reduce((
                data['accept'][:],
                data['hess_struct_update_min_ev'][:] == 0.0,
                data['hess_struct_update_max_ev'][:] == 0.0,
            )).sum() / np.sum(data['accept'][:]),
            'frac_no_tr_update_int_sol': np.logical_and.reduce((
                data['tr_ratio'][:] > 0.75,
                data['iterations_since_tr_update'][:] > 0,
            )).sum() / data['fval'].size,
            'frac_no_tr_update_tr_ratio': np.logical_and.reduce((
                data['tr_ratio'][:] < 0.75,
                data['tr_ratio'][:] > 0.25,
                data['iterations_since_tr_update'][:] > 0,
            )).sum() / data['fval'].size,
            'frac_streak_no_tr_update_tr_ratio':  max_streak(
                np.logical_and.reduce((
                    data['tr_ratio'][:] < 0.75,
                    data['tr_ratio'][:] > 0.25,
                    data['iterations_since_tr_update'][:] > 0,
                )
            )) / data['fval'].size,
            'max_frac_no_tr_update_tr_ratio': np.max(
                np.cumsum(np.logical_and.reduce((
                    data['tr_ratio'][:] < 0.75,
                    data['tr_ratio'][:] > 0.25,
                    data['iterations_since_tr_update'][:] > 0,
                ))) / np.arange(1, data['fval'].size + 1)
            ),
            'max25_frac_no_tr_update_tr_ratio': np.max(
                np.cumsum(np.logical_and.reduce((
                    data['tr_ratio'][:] < 0.75,
                    data['tr_ratio'][:] > 0.25,
                    data['iterations_since_tr_update'][:] > 0,
                )))[25:] / np.arange(26, data['fval'].size + 1)
            ) if data['fval'].size > 25 else 0,
            'max_hess_ev': np.log10(np.min(data['hess_max_ev'][:])),
            'frac_neg_ev': np.sum(data['hess_min_ev'][:] <
                                  -np.sqrt(np.spacing(1))*data['hess_max_ev'])
                / data['fval'].size,
            'frac_singular_shess':
                np.sum(data['cond_shess'][:] > 1 / np.spacing(1)) /
                data['fval'].size,
            'frac_posdef_newt': np.sum(data['posdef'][:]) /
                np.sum(data['step_type'][:] == b'2d'),
            'frac_degenerate_subspace': np.logical_and.reduce((
                data['subspace_dim'][:] == 1,
                np.logical_not(data['newton'][:]),
                data['step_type'][:] == b'2d',
            )).sum() / np.sum(data['step_type'][:] == b'2d'),
            'frac_newton_steps': np.logical_and(
                data['newton'][:],
                data['step_type'][:] == b'2d',
            ).sum() / np.sum(data['step_type'][:] == b'2d'),
            'frac_gradient_steps': np.sum(data['step_type'][:] == b'g') /
                data['fval'].size,
            'frac_border_steps': np.sum(np.logical_and(
                data['step_type'][:] != b'2d',
                data['step_type'][:] != b'nd',
            )) /
                data['fval'].size,
            'converged': np.min(data['fval'][:]) < fmin + CONVERGENCE_THRESHOLD
        } for data in f.values()])
    return stats


for analysis, algos in ANALYSIS_ALGOS.items():
    stats = [
        read_stats(model, opt)
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
                 value_vars=['frac_accepted',
                             'frac_no_hess_update',
                             'frac_no_hess_update_internal',
                             'frac_no_hess_update_border',
                             'frac_no_tr_update_int_sol',
                             'frac_no_tr_update_tr_ratio',
                             'frac_streak_no_tr_update_tr_ratio',
                             'max_frac_no_tr_update_tr_ratio',
                             'max25_frac_no_tr_update_tr_ratio',
                             'frac_no_hess_struct_update',
                             'frac_singular_shess',
                             'frac_neg_ev',
                             'frac_posdef_newt',
                             'frac_degenerate_subspace',
                             'frac_newton_steps',
                             'frac_gradient_steps',
                             'frac_border_steps'])

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
    ).set(xscale='log')
    plt.tight_layout()
    plt.savefig(os.path.join(
        'evaluation',
        f'stats_analysis_{analysis}.pdf'
    ))
    print(f'{analysis} done.')


