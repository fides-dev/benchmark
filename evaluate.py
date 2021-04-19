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
from pypesto.visualize import waterfall, create_references
from pypesto.objective.history import CsvHistory
from pypesto.optimize.result import OptimizerResult
from matplotlib import cm
from compile_petab import load_problem
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
        'fides.subspace=2D.hessian=BFGS',
        'fides.subspace=full.hessian=SR1',
        'fides.subspace=2D.hessian=FIMe',
        'fmincon', 'lsqnonlin', 'ls_trf', 'ls_trf_2D',
    ])
}

OPTIMIZER_FORWARD = [
    'ls_trf',
]

N_STARTS_FORWARD = ['1000']

OPTIMIZER_ADJOINT = ['fides.subspace=full.hessian=BFGS',
                     'fides.subspace=2D.hessian=BFGS',
                     'fides.subspace=full.hessian=SR1',
                     'fides.subspace=2D.hessian=SR1',
                     'ipopt']

N_STARTS_ADJOINT = ['100']

ANALYSIS_ALGOS = {
    'matlab': [x for x in ALGO_COLORS
               if x not in ['ipopt',  'fides.subspace=2D.hessian=SR1',
                            'fides.subspace=full.hessian=BFGS']],
    'curv': ['fides.subspace=2D',
             'fides.subspace=full',
             'fides.subspace=2D.hessian=SR1',
             'fides.subspace=full.hessian=SR1'],
    'hybridB': ['fides.subspace=2D',
                'fides.subspace=2D.hessian=HybridB_5',
                'fides.subspace=2D.hessian=HybridB_2',
                'fides.subspace=2D.hessian=HybridB_1',
                'fides.subspace=2D.hessian=HybridB_05',
                'fides.subspace=2D.hessian=BFGS'],
    'hybridS': ['fides.subspace=2D',
                'fides.subspace=2D.hessian=HybridS_5',
                'fides.subspace=2D.hessian=HybridS_2',
                'fides.subspace=2D.hessian=HybridS_1',
                'fides.subspace=2D.hessian=HybridS_05',
                'fides.subspace=2D.hessian=SR1'],
    'hybridB0': ['fides.subspace=2D',
                 'fides.subspace=2D.hessian=HybridB0_5',
                 'fides.subspace=2D.hessian=HybridB0_2',
                 'fides.subspace=2D.hessian=HybridB0_1',
                 'fides.subspace=2D.hessian=HybridB0_05',
                 'fides.subspace=2D.hessian=BFGS'],
    'hybridS0': ['fides.subspace=2D',
                 'fides.subspace=2D.hessian=HybridS0_5',
                 'fides.subspace=2D.hessian=HybridS0_2',
                 'fides.subspace=2D.hessian=HybridS0_1',
                 'fides.subspace=2D.hessian=HybridS0_05',
                 'fides.subspace=2D.hessian=SR1'],
    'stepback': ['fides.subspace=2D.stepback=reflect_single',
                 'fides.subspace=2D']
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


hass_alias = {
    'Crauste_CellSystems2017': 'Crauste_ImmuneCells_CellSystems2017',
    'Beer_MolBioSystems2014': 'Beer_MolBiosyst2014',
}

matlab_alias = {
    'fmincon': 'trust',
    'lsqnonlin': 'lsq'
}


def load_results_from_benchmark(model, optimizer):
    petab_problem, problem = load_problem(model)

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

    par_idx = np.array([
        par_names.index(par)
        if par in par_names
        else par_names.index(
            par.replace('sigma', 'noise').replace('AKT', 'Akt').replace(
                'scaling', 'scaleFactor').replace('_tot', '')
        )
        for par in [
            petab_problem.x_ids[ix] for ix in petab_problem.x_free_indices
        ]
    ])
    hass2019_chis = np.genfromtxt(os.path.join(
        'Hass2019',
        f'{hass_alias.get(model, model)}_{optimizer}_chi2s.csv',
    ), delimiter=',')
    hass2019_iter = np.genfromtxt(os.path.join(
        'Hass2019',
        f'{hass_alias.get(model, model)}_{optimizer}_iter.csv',
    ), delimiter=',')
    hass2019_ps = np.genfromtxt(os.path.join(
        'Hass2019',
        f'{hass_alias.get(model, model)}_{optimizer}_ps.csv',
    ), delimiter=',')
    if model == 'Fujita_SciSignal2010':
        hass2019_ps = hass2019_ps[:, :19]

    fvals_file = os.path.join(
        'Hass2019',
        f'{hass_alias.get(model, model)}_{optimizer}_fvals.csv',
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

    hass_2019 = pd.read_excel(os.path.join(
        'Hass2019', f'{MODEL_NAME}.xlsx'
    ), sheet_name='Parameters')
    hass_2019.parameter = hass_2019.parameter.apply(
        lambda x: re.sub(r'log10\(([\w_]+)\)', r'\1', x)
    )
    if MODEL_NAME == 'Weber_BMC2015':
        par_df = petab_problem.parameter_df.reset_index().loc[
            [32, 33, 34, 35, 36, 37, 38, 39],
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
        par_df.value = par_df.value.apply(np.log10)
        hass_2019 = hass_2019.append(par_df)

    hass_2019_x = dict(hass_2019[['parameter', 'value']].values)
    x_ref = np.array([
        hass_2019_x.get(
            par, hass_2019_x.get(
                par.replace('sigma', 'noise').replace('AKT', 'Akt').replace(
                    'scaling', 'scaleFactor').replace('_tot', ''),
                None
            )
        )
        for par in petab_problem.x_ids
    ])

    os.makedirs('evaluation', exist_ok=True)

    all_results = []

    refs = create_references(
        x=x_ref[np.asarray(
            petab_problem.x_free_indices
        )],
        fval=problem.objective(
            x_ref[np.asarray(petab_problem.x_free_indices)]),
        legend='Hass2019',
    ) + create_references(
        x=np.asarray(petab_problem.x_nominal_scaled)[np.asarray(
            petab_problem.x_free_indices
        )],
        fval=problem.objective(np.asarray(petab_problem.x_nominal_scaled)[
                                   np.asarray(petab_problem.x_free_indices)]
                               ),
        legend='Hass2019',
    )

    if EVALUATION_TYPE == 'forward':
        optimizers = ['lsqnonlin', 'fmincon'] + OPTIMIZER_FORWARD
        n_starts = N_STARTS_FORWARD[0]
    else:
        optimizers = OPTIMIZER_ADJOINT
        n_starts = N_STARTS_ADJOINT[0]

    for optimizer in optimizers:
        try:
            result = load_results(MODEL_NAME, optimizer, n_starts)
            result.problem = problem
            all_results.append({
                'result': result, 'model': MODEL_NAME, 'optimizer': optimizer,
            })
        except FileNotFoundError:
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

    if EVALUATION_TYPE == 'forward':
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
                palette = 'tab20'
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
            g.set_ylim([0, 4])
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
            'FIMvsHybrid_05_2D': ("fval fides.subspace=2D",
                                  "fval fides.subspace=2D.hessian=HybridB_05"),
            'FIMvsHybrid_2_2D': ("fval fides.subspace=2D",
                                 "fval fides.subspace=2D.hessian=HybridB_2"),
            'FIMvsHybrid_5_2D': ("fval fides.subspace=2D",
                                 "fval fides.subspace=2D.hessian=HybridB_5"),
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

    if EVALUATION_TYPE == 'adjoint':
        opt0 = 'ipopt'
        opt1 = 'fides.subspace=full.hessian=SR1'
        result0 = next(
            r['result'] for r in all_results
            if r['optimizer'] == opt0
        )
        result1 = next(
            r['result'] for r in all_results
            if r['optimizer'] == opt1
        )
        fig, axes = plt.subplots(1, 2)
        fval_offset = np.min([
            np.min(result0.optimize_result.get_for_key('fval')),
            np.min(result1.optimize_result.get_for_key('fval'))
        ]) - 1
        alpha = 0.1
        for start0 in result0.optimize_result.list:
            start1 = next(
                (s for s in result1.optimize_result.list
                 if s['id'] == start0['id']),
                None
            )
            if start1 is None:
                continue
            history0 = CsvHistory(
                file=os.path.join('results',
                                  f'{MODEL_NAME}__{opt0}__100__'
                                  f'trace{start0["id"]}.csv'),
                load_from_file=True
            )
            history1 = CsvHistory(
                file=os.path.join('results',
                                  f'{MODEL_NAME}__{opt1}__100__'
                                  f'trace{start1["id"]}.csv'),
                load_from_file=True
            )
            fvals0 = history0.get_fval_trace() - fval_offset
            fvals1 = history1.get_fval_trace() - fval_offset
            times0 = history0.get_time_trace()
            times1 = history1.get_time_trace()
            i0 = 0
            i1 = 0
            ttrace0 = [start0['fval0'] - fval_offset]
            ttrace1 = [start1['fval0'] - fval_offset]
            if times0[i0] < times1[i1]:
                i0 += 1
            else:
                i1 += 1
            while i0 < len(fvals0) or i1 < len(fvals1):
                ttrace0.append(np.min(fvals0[:np.min([i0+1,
                                                      len(fvals0)-1]) + 1]))
                ttrace1.append(np.min(fvals1[:np.min([i1+1,
                                                      len(fvals1)-1]) + 1]))
                if i1 == len(fvals1) or (not(i0 == len(fvals0)) and
                                         times0[i0] < times1[i1]):
                    i0 += 1
                else:
                    i1 += 1
            for ax in axes:
                ax.plot(ttrace0, ttrace1, 'k.-', alpha=alpha)
                ax.plot(ttrace0[-1], ttrace1[-1], 'r.', zorder=99)

        for iax, ax in enumerate(axes):
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_aspect('equal')

            xy_min = np.min([ax.get_xlim()[0], ax.get_ylim()[0]])
            if iax == 0:
                xy_max = np.max([ax.get_xlim()[1], ax.get_ylim()[1]])
            else:
                xy_max = 1e8

            ax.set_xlim([xy_min, xy_max])
            ax.set_ylim([xy_min, xy_max])
            ax.set_xlabel(f'funcion value {opt0}')
            ax.set_ylabel(f'funcion value {opt1}')
            ax.plot([xy_min, xy_max], [xy_min, xy_max], 'k:')

        plt.tight_layout()
        plt.savefig(os.path.join(
            'evaluation', f'{MODEL_NAME}_{EVALUATION_TYPE}.pdf')
        )
