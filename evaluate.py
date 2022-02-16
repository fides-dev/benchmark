import sys
import os
import re
import petab
import pypesto
import h5py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

from pypesto.store import OptimizationResultHDF5Reader
from pypesto.visualize import waterfall
from pypesto.result import OptimizerResult
from matplotlib import cm
from compile_petab import load_problem, PARAMETER_ALIASES, MODEL_ALIASES
from benchmark import set_solver_model_options


def get_stats_file(model_name, optimizer):
    return os.path.join(
        'stats',
        f'{model_name}__{optimizer}__{N_STARTS_FORWARD[0]}__STATS.hdf5'
    )


new_rc_params = {
    "font.family": 'Helvetica',
    "pdf.fonttype": 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
}
mpl.rcParams.update(new_rc_params)

CONVERGENCE_THRESHOLDS = [0.05, 1, 5]

OPTIMIZER_FORWARD = [
    'fides.subspace=2D',
    'fides.subspace=full',
    'fides.subspace=2D.hessian=HybridB_25',
    'fides.subspace=2D.hessian=HybridB_50',
    'fides.subspace=2D.hessian=HybridB_75',
    'fides.subspace=2D.hessian=HybridB_100',
    'fides.subspace=full.hessian=BFGS',
    'fides.subspace=2D.hessian=BFGS',
    'fides.subspace=full.hessian=SR1',
    'fides.subspace=2D.hessian=SR1',
    'fides.subspace=2D.hessian=FX',
    'fides.subspace=2D.hessian=SSM',
    'fides.subspace=2D.hessian=TSSM',
    'fides.subspace=2D.hessian=GNSBFGS',
    'fides.subspace=2D.stepback=reflect_single',
    'fides.subspace=2D.ebounds=10',
    'fides.subspace=2D.ebounds=100',
    'fides.subspace=2D.ebounds=Inf',
    'ls_trf_2D',
    'fides.subspace=2D.hessian=FIMe',
]

N_STARTS_FORWARD = ['1000']

ANALYSIS_ALGOS = {
    'matlab': [
        'fides.subspace=2D',
        'fides.subspace=2D.hessian=FIMe',
        'fmincon',
        'lsqnonlin',
        'ls_trf_2D',
    ],
    'stepback': [
        'fides.subspace=2D',
        'fides.subspace=2D.stepback=reflect_single',
        'fides.subspace=2D.ebounds=10',
        'fides.subspace=2D.ebounds=100',
        'fides.subspace=2D.ebounds=Inf',
    ],
    'curv': [
        'fides.subspace=2D',
        'fides.subspace=full',
        'fides.subspace=2D.hessian=BFGS',
        'fides.subspace=full.hessian=BFGS',
        'fides.subspace=2D.hessian=SR1',
        'fides.subspace=full.hessian=SR1'
    ],
    'hybridB': [
        'fides.subspace=2D',
        'fides.subspace=2D.hessian=HybridB_100',
        'fides.subspace=2D.hessian=HybridB_75',
        'fides.subspace=2D.hessian=HybridB_50',
        'fides.subspace=2D.hessian=HybridB_25',
        'fides.subspace=2D.hessian=BFGS'
    ],
    'hybrid': [
        'fides.subspace=2D',
        'fides.subspace=2D.hessian=HybridB_50',
        'fides.subspace=2D.hessian=FX',
        'fides.subspace=2D.hessian=SSM',
        'fides.subspace=2D.hessian=TSSM',
        'fides.subspace=2D.hessian=GNSBFGS'
    ],
}

ALGO_PALETTES = {
    'matlab': 'tab10',
    'curv': 'tab20',
    'hybrid': 'Dark2',
    'hybridB': 'Blues',
    'stepback': 'Set2',
}

ALGO_COLORS = {
    analysis: {
        legend: tuple([*cm.get_cmap(cmap).colors[il], 1.0])
        if cmap not in ['Blues']
        else tuple(cm.get_cmap('Blues')(
            range(len(ANALYSIS_ALGOS[analysis])))[il]
        )
        for il, legend in enumerate(ANALYSIS_ALGOS[analysis])
    }
    for analysis, cmap in ALGO_PALETTES.items()
}


def get_num_converged(fvals, fmin, threshold=CONVERGENCE_THRESHOLDS[1]):
    return np.nansum(np.asarray(fvals) < fmin + threshold)


def get_num_converged_per_grad(fvals, n_grads, fmin,
                               threshold=CONVERGENCE_THRESHOLDS[1]):
    return get_num_converged(fvals, fmin, threshold) / np.nansum(n_grads)


def load_results(model, optimizer, n_starts):
    if optimizer in ['lsqnonlin', 'fmincon']:
        return load_results_from_benchmark(model, optimizer)
    else:
        return load_results_from_hdf5(model, optimizer, n_starts)


def load_results_from_hdf5(model, optimizer, n_starts):
    file = f'{model}__{optimizer}__{n_starts}.hdf5'
    path = os.path.join('results', file)
    if os.path.exists(path):
        reader = OptimizationResultHDF5Reader(path)
        print(f'Loaded results from {file}')
        return reader.read()

    result = pypesto.Result()
    stats_file = get_stats_file(model, optimizer)
    result.optimize_result = pypesto.OptimizeResult()
    with h5py.File(stats_file, 'r') as f:
        result.optimize_result.list = [OptimizerResult(**{
            'fval': np.min(data['fval'][:]),
            'n_fval': data['fval'].size+1,
            'n_grad': data['fval'].size+1,
            'n_hess': data['fval'].size+1,
            'n_res': 0,
            'n_sres': 0,
            'id': str(idx),
        }) for idx, data in enumerate(f.values())]
    result.optimize_result.sort()
    print(f'Loaded incomplete results from {stats_file} ('
          f'{len(result.optimize_result.list)}/{n_starts})')
    return result


matlab_alias = {
    'fmincon': 'trust',
    'lsqnonlin': 'lsq'
}


def load_results_from_benchmark(model, optimizer):
    petab_problem, problem = load_problem(model)
    if isinstance(problem.objective, pypesto.AmiciObjective):
        objective = problem.objective
    else:
        objective = problem.objective._objectives[0]
    set_solver_model_options(objective.amici_solver,
                             objective.amici_model)

    hass_2019_pars = pd.read_excel(os.path.join(
        'Hass2019', f'{model}.xlsx'
    ), sheet_name='Parameters')
    hass_2019_pars.parameter = hass_2019_pars.parameter.apply(
        lambda x: re.sub(r'log10\(([\w_]+)\)', r'\1', x)
    )

    if model == 'Weber_BMC2015':
        hass_2019_pars = hass_2019_pars.append(
            petab_problem.parameter_df.reset_index().loc[
                [39, 34, 38, 33, 35, 36, 37, 32],
                [petab.PARAMETER_ID, petab.NOMINAL_VALUE,
                 petab.LOWER_BOUND,
                 petab.UPPER_BOUND, petab.PARAMETER_SCALE,
                 petab.ESTIMATE]].rename(
                columns={
                    petab.PARAMETER_ID: 'parameter',
                    petab.NOMINAL_VALUE: 'value',
                    petab.LOWER_BOUND: 'lower boundary',
                    petab.UPPER_BOUND: 'upper boundary',
                    petab.PARAMETER_SCALE: 'analysis at log-scale',
                    petab.ESTIMATE: 'estimated'
                }
            )
        )

    par_names = list(hass_2019_pars.parameter)

    palias = PARAMETER_ALIASES.get(model, {})

    par_idx = np.array([
        par_names.index(palias.get(par, par))
        for par in [
            petab_problem.x_ids[ix] for ix in petab_problem.x_free_indices
        ]
    ])
    hass_model = MODEL_ALIASES.get(model, model)
    hass2019_chis = np.genfromtxt(os.path.join(
        'Hass2019',
        f'{hass_model}_{optimizer}_chi2s.csv',
    ), delimiter=',')
    hass2019_iter = np.genfromtxt(os.path.join(
        'Hass2019',
        f'{hass_model}_{optimizer}_iter.csv',
    ), delimiter=',')
    hass2019_ps = np.genfromtxt(os.path.join(
        'Hass2019',
        f'{hass_model}_{optimizer}_ps.csv',
    ), delimiter=',')
    if model == 'Fujita_SciSignal2010':
        hass2019_ps = hass2019_ps[:, :19]

    fvals_file = os.path.join(
        'Hass2019',
        f'{hass_model}_{optimizer}_fvals.csv',
    )
    if os.path.exists(fvals_file):
        hass2019_fvals = np.genfromtxt(fvals_file, delimiter=',')
    else:
        hass2019_fvals = np.array([
            problem.objective(p[par_idx])
            if not np.isnan(p).any() else np.NaN
            for p in hass2019_ps
        ])
        assert np.isfinite(hass2019_fvals[np.nanargmin(hass2019_chis)])
        assert np.abs(hass2019_chis[np.nanargmin(hass2019_chis)] -
                      hass2019_chis[np.nanargmin(hass2019_fvals)]) < 0.05
        np.savetxt(
            fvals_file, hass2019_fvals, delimiter=','
        )

    result = pypesto.Result()
    result.optimize_result = pypesto.OptimizeResult()
    for id, (fval, p, n_grad) in enumerate(zip(hass2019_fvals, hass2019_ps,
                                               hass2019_iter)):
        if not np.isfinite(fval):
            continue
        result.optimize_result.append(OptimizerResult(
            id=str(id),
            x=p,
            fval=fval,
            n_grad=n_grad,
            n_sres=0,
        ))
    result.optimize_result.sort()

    print(f'Loaded results from {fvals_file}.')
    return result


if __name__ == '__main__':
    MODEL_NAME = sys.argv[1]

    petab_problem, problem = load_problem(MODEL_NAME)
    set_solver_model_options(problem.objective.amici_solver,
                             problem.objective.amici_model)

    os.makedirs('evaluation', exist_ok=True)

    all_results = []

    optimizers = OPTIMIZER_FORWARD + ['fmincon', 'lsqnonlin']
    n_starts = N_STARTS_FORWARD[0]

    for optimizer in optimizers:
        try:
            result = load_results(MODEL_NAME, optimizer, n_starts)
            result.problem = problem
            all_results.append({
                'result': result, 'optimizer': optimizer,
            })
        except (FileNotFoundError, IOError):
            pass

    all_results = sorted(
        all_results,
        key=lambda r: r['result'].optimize_result.list[0]['fval']
    )

    waterfall_results = [r for r in all_results
                         if r['optimizer'] in ALGO_COLORS]

    fmin = np.nanmin([r['result'].optimize_result.list[0]['fval']
                      for r in all_results])

    waterfall(
        [r['result'] for r in waterfall_results],
        legends=[r['optimizer'] for r in waterfall_results],
        colors=[ALGO_COLORS[r['optimizer']] for r in waterfall_results],
        size=(4.25, 3.5),
    )
    plt.tight_layout()
    plt.savefig(os.path.join('evaluation',
                             f'{MODEL_NAME}_all_starts.pdf'))

    waterfall(
        [r['result'] for r in waterfall_results],
        legends=[r['optimizer'] for r in waterfall_results],
        colors=[ALGO_COLORS[r['optimizer']] for r in waterfall_results],
        start_indices=range(int(int(n_starts)/10)),
        size=(4.25, 3.5),
    )
    plt.tight_layout()
    plt.savefig(os.path.join(
        'evaluation',
        f'{MODEL_NAME}_{int(int(n_starts)/10)}_starts.pdf'
    ))

    waterfall_results_stepback = [
        r for r in all_results if r['optimizer'] in ANALYSIS_ALGOS['stepback']
    ]

    waterfall(
        [r['result'] for r in waterfall_results_stepback],
        legends=[r['optimizer'] for r in waterfall_results_stepback],
        colors=[
            tuple([*c, 1.0]) for c in sns.color_palette(
                ALGO_PALETTES['stepback'], len(waterfall_results_stepback)
            )
        ],
        start_indices=range(int(int(n_starts) / 10)),
        size=(4, 3.5),
    )
    plt.tight_layout()
    plt.savefig(os.path.join(
        'evaluation',
        f'{MODEL_NAME}_{int(int(n_starts) / 10)}_starts_stepback.pdf'
    ))

    df = pd.DataFrame([
        {
            'fval': start['fval'],
            'iter': start['n_grad'] + start['n_sres'],
            'id': start['id'],
            'optimizer': results['optimizer']
        }
        for results in all_results
        for start in results['result'].optimize_result.list
        if results['optimizer'] in OPTIMIZER_FORWARD + ['fmincon',
                                                        'lsqnonlin']
    ])

    df['opt_subspace'] = df['optimizer'].apply(
        lambda x:
        'fides ' + x.split('.')[1].split('=')[1]
        if len(x.split('.')) > 1
        else 'ls_trf full'
    )

    df['hessian'] = df['optimizer'].apply(
        lambda x: x.split('.')[2].split('=')[1] if len(x.split('.')) > 2
        else 'FIM'
    )

    for analysis, algos in ANALYSIS_ALGOS.items():

        palette = ALGO_PALETTES[analysis]

        plt.subplots()
        g = sns.boxplot(
            data=df,
            order=algos,
            palette=palette,
            x='optimizer', y='iter',
            log=True,
        )
        g.set_xticklabels(g.get_xticklabels(), rotation=90)
        g.set(yscale='log', ylim=[1e0, 1e5])
        plt.tight_layout()
        plt.savefig(os.path.join(
            'evaluation',
            f'{MODEL_NAME}_iter_{analysis}.pdf'
        ))

    df_pivot = df[
        df.optimizer.apply(lambda x: x in OPTIMIZER_FORWARD)
    ].pivot(index='id', columns=['optimizer'])

    df_pivot.fval = np.log10(df_pivot.fval - fmin + 1)

    df_pivot = df_pivot[(df_pivot.fval < 1e5).all(axis=1)]

    df_pivot.columns = [' '.join(col).strip()
                        for col in df_pivot.columns.values]

    for optimizer in optimizers:
        if optimizer == 'fval fides.subspace=2D':
            continue

        x = 'fval fides.subspace=2D'
        y = f'fval {optimizer}'
        if y not in df_pivot:
            continue
        lb, ub = [
            fun([fun(df_pivot[x]), fun(df_pivot[y])])
            for fun in [np.nanmin, np.nanmax]
        ]
        lb -= (ub - lb) / 10
        ub += (ub - lb) / 10

        sns.jointplot(
            data=df_pivot,
            x=x,
            y=y,
            kind='scatter', xlim=(lb, ub), ylim=(lb, ub),
            alpha=0.3,
            marginal_kws={'bins': 25},
        )
        plt.tight_layout()
        plt.savefig(os.path.join(
            'evaluation',
            f'{MODEL_NAME}_fval_{optimizer}.pdf'
        ))
