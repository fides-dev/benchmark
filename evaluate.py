import sys
import os
import re
import petab
import pypesto
import sklearn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

from pypesto.store import OptimizationResultHDF5Reader
from pypesto.visualize import waterfall
from pypesto.optimize.result import OptimizerResult
from matplotlib import cm
from compile_petab import load_problem, PARAMETER_ALIASES, MODEL_ALIASES
from benchmark import set_solver_model_options

new_rc_params = {
    "font.family": 'Helvetica',
    "pdf.fonttype": 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
}
mpl.rcParams.update(new_rc_params)

CONVERGENCE_THRESHOLD = 0.05

cmap = cm.get_cmap('tab10')
ALGO_COLORS = {
    legend: tuple([*cmap.colors[il], 1.0])
    for il, legend in enumerate([
        'fides.subspace=2D',
        'fmincon', 'lsqnonlin', 'ls_trf_2D',
    ])
}

OPTIMIZER_FORWARD = [
    'fides.subspace=2D',
    'fides.subspace=full',
    'fides.subspace=2D.hessian=HybridB_5',
    'fides.subspace=2D.hessian=HybridB_10',
    'fides.subspace=2D.hessian=HybridB_15',
    'fides.subspace=2D.hessian=HybridB_20',
    'fides.subspace=2D.hessian=HybridB_25',
    'fides.subspace=full.hessian=BFGS',
    'fides.subspace=2D.hessian=BFGS',
    'fides.subspace=full.hessian=SR1',
    'fides.subspace=2D.hessian=SR1',
    'fides.subspace=full.hessian=FX',
    'fides.subspace=2D.hessian=FX',
    'fides.subspace=full.hessian=SSM',
    'fides.subspace=2D.hessian=SSM',
    'fides.subspace=full.hessian=TSSM',
    'fides.subspace=2D.hessian=TSSM',
    'fides.subspace=full.hessian=GNSBFGS',
    'fides.subspace=2D.hessian=GNSBFGS',
    'fides.subspace=2D.stepback=reflect_single',
    'fides.subspace=2D.ebounds=True',
    'ls_trf',
    'ls_trf_2D',
    'fides.subspace=2D.hessian=FIMe',
]

N_STARTS_FORWARD = ['1000']

ANALYSIS_ALGOS = {
    'matlab': [x for x in ALGO_COLORS
               if x not in ['ipopt',  'fides.subspace=2D.hessian=SR1',
                            'fides.subspace=full.hessian=BFGS']],
    'curv': ['fides.subspace=2D',
             'fides.subspace=full',
             'fides.subspace=2D.hessian=BFGS',
             'fides.subspace=full.hessian=BFGS',
             'fides.subspace=2D.hessian=SR1',
             'fides.subspace=full.hessian=SR1'],
    'hybridB': ['fides.subspace=2D',
                'fides.subspace=2D.hessian=HybridB_25',
                'fides.subspace=2D.hessian=HybridB_20',
                'fides.subspace=2D.hessian=HybridB_15',
                'fides.subspace=2D.hessian=HybridB_10',
                'fides.subspace=2D.hessian=HybridB_5',
                'fides.subspace=2D.hessian=BFGS'],
    'hybridS': ['fides.subspace=2D',
                'fides.subspace=2D.hessian=HybridS_25',
                'fides.subspace=2D.hessian=HybridS_20',
                'fides.subspace=2D.hessian=HybridS_15',
                'fides.subspace=2D.hessian=HybridS_10',
                'fides.subspace=2D.hessian=HybridS_5',
                'fides.subspace=2D.hessian=SR1'],
    'hybridB0': ['fides.subspace=2D',
                 'fides.subspace=2D.hessian=HybridB0_25',
                 'fides.subspace=2D.hessian=HybridB0_20',
                 'fides.subspace=2D.hessian=HybridB0_15',
                 'fides.subspace=2D.hessian=HybridB0_10',
                 'fides.subspace=2D.hessian=HybridB0_5',
                 'fides.subspace=2D.hessian=BFGS'],
    'hybridS0': ['fides.subspace=2D',
                 'fides.subspace=2D.hessian=HybridS0_25',
                 'fides.subspace=2D.hessian=HybridS0_20',
                 'fides.subspace=2D.hessian=HybridS0_15',
                 'fides.subspace=2D.hessian=HybridS0_10',
                 'fides.subspace=2D.hessian=HybridS0_5',
                 'fides.subspace=2D.hessian=SR1'],
    'hybrid': ['fides.subspace=2D',
               'fides.subspace=2D.hessian=FX',
               'fides.subspace=full.hessian=FX',
               'fides.subspace=2D.hessian=SSM',
               'fides.subspace=full.hessian=SSM',
               'fides.subspace=2D.hessian=TSSM',
               'fides.subspace=full.hessian=TSSM',
               'fides.subspace=2D.hessian=GNSBFGS',
               'fides.subspace=full.hessian=GNSBFGS'],
    'stepback': ['fides.subspace=2D.stepback=reflect_single',
                 'fides.subspace=2D',
                 'fides.subspace=2D.ebounds=True']
}


def get_num_converged(fvals, fmin):
    return np.nansum(np.asarray(fvals) < fmin + CONVERGENCE_THRESHOLD)


def get_num_converged_per_grad(fvals, n_grads, fmin):
    return get_num_converged(fvals, fmin) / np.nansum(n_grads)


def get_dist(fvals, fmin):
    fvals = np.asarray(fvals)
    converged_fvals = fvals[fvals < fmin + CONVERGENCE_THRESHOLD]
    if len(converged_fvals) < 2:
        return 0
    distances = sklearn.metrics.pairwise_distances(
        converged_fvals.reshape(-1, 1)
    )
    distances[np.diag_indices(len(converged_fvals))] = np.Inf
    return 1/np.median(distances.min(axis=1))


def load_results(model, optimizer, n_starts):
    if optimizer in ['lsqnonlin', 'fmincon']:
        return load_results_from_benchmark(model, optimizer)
    else:
        return load_results_from_hdf5(model, optimizer, n_starts)


def load_results_from_hdf5(model, optimizer, n_starts):
    file = f'{model}__{optimizer}__{n_starts}.hdf5'
    reader = OptimizationResultHDF5Reader(os.path.join('results', file))
    return reader.read()


matlab_alias = {
    'fmincon': 'trust',
    'lsqnonlin': 'lsq'
}


def load_results_from_benchmark(model, optimizer):
    petab_problem, problem = load_problem(model)
    set_solver_model_options(problem.objective.amici_solver,
                             problem.objective.amici_model)

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
    return result


if __name__ == '__main__':
    MODEL_NAME = sys.argv[1]
    EVALUATION_TYPE = sys.argv[2]

    petab_problem, problem = load_problem(MODEL_NAME)
    set_solver_model_options(problem.objective.amici_solver,
                             problem.objective.amici_model)

    os.makedirs('evaluation', exist_ok=True)

    all_results = []

    optimizers = ['lsqnonlin', 'fmincon'] + OPTIMIZER_FORWARD
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
                             f'{MODEL_NAME}_all_starts_{EVALUATION_TYPE}.pdf'))

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
        f'{MODEL_NAME}_{int(int(n_starts)/10)}_starts_{EVALUATION_TYPE}.pdf'
    ))

    waterfall_results_stepback = [
        r for r in all_results if r['optimizer'] in ANALYSIS_ALGOS['stepback']
    ]

    waterfall(
        [r['result'] for r in waterfall_results_stepback],
        legends=[r['optimizer'] for r in waterfall_results_stepback],
        colors=[
            tuple([*c, 1.0]) for c in sns.color_palette('Set2', 2)
        ],
        start_indices=range(int(int(n_starts) / 10)),
        size=(4, 3.5),
    )
    plt.tight_layout()
    plt.savefig(os.path.join(
        'evaluation',
        f'{MODEL_NAME}_{int(int(n_starts) / 10)}_starts_'
        f'{EVALUATION_TYPE}_stepback.pdf'
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

    df.iter = df.iter.apply(np.log10)

    for analysis, algos in ANALYSIS_ALGOS.items():

        if analysis == 'matlab':
            palette = [
                ALGO_COLORS.get(algo, ALGO_COLORS.get('ipopt'))
                for algo in algos
            ]
        elif analysis == 'curv':
            palette = 'Dark2'
        elif analysis in ['hybridB', 'hybridS', 'hybridB0', 'hybridS0']:
            palette = 'Blues'
        elif analysis == 'stepback':
            palette = 'Set2'

        plt.subplots()
        g = sns.boxplot(
            data=df,
            order=algos,
            palette=palette,
            x='optimizer', y='iter'
        )
        g.set_xticklabels(g.get_xticklabels(), rotation=90)
        g.set_ylim([0, 5])
        plt.tight_layout()
        plt.savefig(os.path.join(
            'evaluation',
            f'{MODEL_NAME}_iter_{analysis}_{EVALUATION_TYPE}.pdf'
        ))

    df_pivot = df[
        df.optimizer.apply(lambda x: x in OPTIMIZER_FORWARD)
    ].pivot(index='id', columns=['optimizer'])

    df_pivot.fval = np.log10(df_pivot.fval - fmin + 1)

    df_pivot = df_pivot[(df_pivot.fval < 1e5).all(axis=1)]

    df_pivot.columns = [' '.join(col).strip()
                        for col in df_pivot.columns.values]

    for name, vals in {
        '2Dvsfull_FIM': ('fval fides.subspace=2D',
                         'fval fides.subspace=full'),
        'FIMvsBFGS_2D': ("fval fides.subspace=2D",
                         "fval fides.subspace=2D.hessian=BFGS"),
        'FIMvsSR1_2D': ("fval fides.subspace=2D",
                        "fval fides.subspace=2D.hessian=SR1"),
        'FIMvsHybrid_5_2D': ("fval fides.subspace=2D",
                              "fval fides.subspace=2D.hessian=HybridB_5"),
        'FIMvsHybrid_25_2D': ("fval fides.subspace=2D",
                             "fval fides.subspace=2D.hessian=HybridB_25"),
        'reflect': ("fval fides.subspace=2D",
                    "fval fides.subspace=2D.stepback=reflect_single"),
    }.items():
        lb, ub = [
            fun([fun(df_pivot[vals[0]]),
                 fun(df_pivot[vals[1]])])
            for fun in [np.nanmin, np.nanmax]
        ]
        lb -= (ub - lb) / 10
        ub += (ub - lb) / 10

        sns.jointplot(
            data=df_pivot,
            x=vals[0],
            y=vals[1],
            kind='scatter', xlim=(lb, ub), ylim=(lb, ub),
            alpha=0.3,
            marginal_kws={'bins': 25},
        )
        plt.tight_layout()
        plt.savefig(os.path.join(
            'evaluation',
            f'{MODEL_NAME}_fval_{name}_{EVALUATION_TYPE}.pdf'
        ))

    df_metrics = pd.DataFrame([
        {
            'convergence_count': get_num_converged(
                results['result'].optimize_result.get_for_key('fval'),
                fmin
            ),
            'conv_per_grad': get_num_converged_per_grad(
                results['result'].optimize_result.get_for_key('fval'),
                results['result'].optimize_result.get_for_key('n_sres')
                if results['optimizer'] == 'ls_tr' else
                results['result'].optimize_result.get_for_key('n_grad'),
                fmin
            ),
            'consistency': get_dist(
                results['result'].optimize_result.get_for_key('fval'),
                fmin
            ),

            'optimizer': results['optimizer']
        }
        for results in all_results
    ])

    df_metrics['opt_subspace'] = df_metrics['optimizer'].apply(
        lambda x:
        'fides ' + x.split('.')[1].split('=')[1]
        if len(x.split('.')) > 1
        else ''
    )

    df_metrics['hessian'] = df_metrics['optimizer'].apply(
        lambda x: x.split('.')[2].split('=')[1] if len(x.split('.')) > 2
        else 'FIM'
    )

    for analysis, algos in ANALYSIS_ALGOS.items():
        for metric in ['convergence_count', 'conv_per_grad', 'consistency']:
            plt.subplots()
            g = sns.barplot(data=df_metrics, x='optimizer', y=metric,
                            order=[x for x in algos])
            plt.tight_layout()
            plt.savefig(os.path.join(
                'evaluation',
                f'{MODEL_NAME}_{metric}_{analysis}_{EVALUATION_TYPE}.pdf'
            ))
