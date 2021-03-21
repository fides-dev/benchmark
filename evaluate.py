import sys
import os
import re
import matplotlib as mpl

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pypesto.store import OptimizationResultHDF5Reader
from pypesto.visualize import waterfall, create_references
from pypesto.objective.history import CsvHistory
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

CONVERGENCE_THRESHOLD = 0.5

cmap = cm.get_cmap('tab10')
ALGO_COLORS = {
    legend: tuple([*cmap.colors[il], 1.0])
    for il, legend in enumerate([
        'fides.subspace=full', 'fides.subspace=2D', 'ls_trf', 'ipopt',
        'fides.subspace=full.hessian=BFGS',
        'fides.subspace=2D.hessian=BFGS', 'Hass2019',
        'fmincon', 'lsqnonlin'
    ])
}


def get_num_converged(fvals, fmin):
    return np.nansum(np.asarray(fvals) < fmin + CONVERGENCE_THRESHOLD)


def get_num_converged_per_grad(fvals, n_grads, fmin):
    return get_num_converged(fvals, fmin) / np.nansum(n_grads)


def load_results_from_hdf5(model, optimizer, n_starts):
    file = f'{model}__{optimizer}__{n_starts}.hdf5'
    reader = OptimizationResultHDF5Reader(os.path.join('results', file))
    return reader.read()


if __name__ == '__main__':
    MODEL_NAME = sys.argv[1]
    EVALUATION_TYPE = sys.argv[2]

    hdf5_files = [r for r in os.listdir('results')
                  if r.startswith(MODEL_NAME) and r.endswith('.hdf5')]

    petab_problem, problem = load_problem(MODEL_NAME)
    set_solver_model_options(problem.objective.amici_solver,
                             problem.objective.amici_model)

    hass_2019 = pd.read_excel(os.path.join(
        'Hass2019', f'{MODEL_NAME}.xlsx'
    ), sheet_name='Parameters')
    hass_2019.parameter = hass_2019.parameter.apply(
        lambda x: re.sub(r'log10\(([\w_]+)\)', r'\1', x)
    )

    hass_2019_x = dict(hass_2019[['parameter', 'value']].values)
    x_ref = np.array([
        hass_2019_x.get(
            par, hass_2019_x.get(
                par.replace('sigma', 'noise').replace('AKT', 'Akt').replace(
                    'scaling', 'scaleFactor').replace('_tot', ''),
                None
        ))
        for par in petab_problem.x_ids
    ])

    hass_alias = {
        'Crauste_CellSystems2017': 'Crauste_ImmuneCells_CellSystems2017',
        'Beer_MolBioSystems2014': 'Beer_MolBiosyst2014',
    }

    hass2019_fmintrust_chis = np.genfromtxt(os.path.join(
        'Hass2019',
        f'{hass_alias.get(MODEL_NAME, MODEL_NAME)}_trust_chi2s.csv',
    ), delimiter=',')

    hass2019_fmintrust_ps = np.genfromtxt(os.path.join(
        'Hass2019',
        f'{hass_alias.get(MODEL_NAME, MODEL_NAME)}_trust_ps.csv'
    ), delimiter=',')

    refs = create_references(
        x=x_ref[np.asarray(
            petab_problem.x_free_indices
        )],
        fval=problem.objective(x_ref[np.asarray(petab_problem.x_free_indices)]),
        legend='Hass2019',
        color=ALGO_COLORS['Hass2019']
    ) + create_references(
        x=hass2019_fmintrust_ps[hass2019_fmintrust_chis.argmin(),
                                np.asarray(petab_problem.x_free_indices)],
        fval=problem.objective(
            hass2019_fmintrust_ps[hass2019_fmintrust_chis.argmin(),
                                  np.asarray(petab_problem.x_free_indices)]
        ),
        legend='Hass2019',
        color=ALGO_COLORS['Hass2019']
    ) + create_references(
        x=np.asarray(petab_problem.x_nominal_scaled)[np.asarray(
            petab_problem.x_free_indices
        )],
        fval=problem.objective(np.asarray(petab_problem.x_nominal_scaled)[
            np.asarray(petab_problem.x_free_indices)]
        ),
        legend='Hass2019',
        color=ALGO_COLORS['Hass2019']
    )

    if MODEL_NAME == 'Fiedler_BMC2016':
        # benchmark results worse than reference
        ref = refs[1]
        ref.fval = -56.86964545865438
    else:
        ref = refs[np.argmin([r.fval for r in refs])]

    os.makedirs('evaluation', exist_ok=True)

    all_results = []
    for hdf_results_file in hdf5_files:
        MODEL, OPTIMIZER, N_STARTS = \
            os.path.splitext(hdf_results_file)[0].split('__')

        if MODEL == MODEL_NAME and (OPTIMIZER != 'ls_trf' or
                                    MODEL in ['Fujita_SciSignal2010',
                                              'Crauste_CellSystems2017']):

            result = load_results_from_hdf5(MODEL, OPTIMIZER, N_STARTS)
            result.problem = problem
            all_results.append({
                'result': result, 'model': MODEL_NAME, 'optimizer': OPTIMIZER,
                'file': hdf_results_file
            })

    all_results = sorted(
        all_results,
        key=lambda r: r['result'].optimize_result.list[0]['fval']
    )

    waterfall_results = [r for r in all_results
                         if r['optimizer'] in ALGO_COLORS]

    fmin = np.nanmin([r['result'].optimize_result.list[0]['fval']
                      for r in all_results] + [ref.fval])

    waterfall(
        [r['result'] for r in waterfall_results],
        reference=ref,
        legends=[r['optimizer'] for r in waterfall_results],
        colors=[ALGO_COLORS[r['optimizer']] for r in waterfall_results],
        size=(6, 3.5),
    )
    plt.tight_layout()
    plt.savefig(os.path.join('evaluation',
                             f'{MODEL_NAME}_all_starts_{EVALUATION_TYPE}.pdf'))

    waterfall(
        [r['result'] for r in waterfall_results],
        reference=ref,
        legends=[r['optimizer'] for r in waterfall_results],
        colors=[ALGO_COLORS[r['optimizer']] for r in waterfall_results],
        start_indices=range(int(int(N_STARTS)/10)),
        size=(6, 3.5),
    )
    plt.tight_layout()
    plt.savefig(os.path.join(
        'evaluation',
        f'{MODEL_NAME}_{int(int(N_STARTS)/10)}_starts_{EVALUATION_TYPE}.pdf'
    ))

    if EVALUATION_TYPE == 'forward':
        df = pd.DataFrame([
            {
                'fval': start['fval'],
                'time': start['time'],
                'iter': start['n_grad'] + start['n_sres'],
                'id': start['id'],
                'optimizer': results['optimizer']
            }
            for results in all_results
            for start in results['result'].optimize_result.list
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

        plt.subplots()
        g = sns.boxplot(data=df, x='hessian', y='iter', hue='opt_subspace',
                        order=['FIM', 'Hybrid_05', 'Hybrid_1', 'Hybrid_2',
                               'Hybrid_5', 'BFGS', 'SR1'],
                        hue_order=['fides 2D', 'fides full', 'ls_trf'],
                        )
        g.set_yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(
            'evaluation',
            f'{MODEL_NAME}_iter_{EVALUATION_TYPE}.pdf'
        ))

        df_pivot = df.pivot(index='id', columns=['optimizer'])

        df_pivot.fval = np.log10(df_pivot.fval - fmin + 1)

        df_pivot = df_pivot[(df_pivot.fval < 1e5).all(axis=1)]

        df_pivot.columns = [' '.join(col).strip()
                            for col in df_pivot.columns.values]

        for name, vals in {
            '2Dvsfull_FIM': ('fval fides.subspace=2D',
                             'fval fides.subspace=full'),
            '2Dvsfull_Hybrid': ("fval fides.subspace=2D.hessian=Hybrid_2",
                                "fval fides.subspace=full.hessian=Hybrid_2"),
            'FIMvsBFGS_2D': ("fval fides.subspace=2D",
                             "fval fides.subspace=2D.hessian=BFGS"),
            'FIMvsSR1_2D': ("fval fides.subspace=2D",
                            "fval fides.subspace=2D.hessian=SR1"),
            'FIMvsHybrid_05_2D': ("fval fides.subspace=2D",
                                  "fval fides.subspace=2D.hessian=Hybrid_05"),
            'FIMvsHybrid_2_2D': ("fval fides.subspace=2D",
                                 "fval fides.subspace=2D.hessian=Hybrid_2"),
            'FIMvsHybrid_5_2D': ("fval fides.subspace=2D",
                                 "fval fides.subspace=2D.hessian=Hybrid_5"),
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

        df_time = pd.DataFrame([
            {
                'convergence_count': get_num_converged(
                    results['result'].optimize_result.get_for_key('fval'),
                    fmin
                ),
                'total_time': np.sum(
                    results['result'].optimize_result.get_for_key('time')
                ),
                'conv_per_grad': get_num_converged_per_grad(
                    results['result'].optimize_result.get_for_key('fval'),
                    results['result'].optimize_result.get_for_key('n_grad'),
                    fmin
                ),
                'optimizer': results['optimizer']
            }
            for results in all_results
        ])

        df_time['opt_subspace'] = df_time['optimizer'].apply(
            lambda x:
            'fides ' + x.split('.')[1].split('=')[1]
            if len(x.split('.')) > 1
            else 'ls_trf full'
        )

        df_time['hessian'] = df_time['optimizer'].apply(
            lambda x: x.split('.')[2].split('=')[1] if len(x.split('.')) > 2
            else 'FIM'
        )

        for metric in ['convergence_count', 'conv_per_grad']:
            plt.subplots()
            g = sns.barplot(data=df_time,
                            x='hessian', y=metric, hue='opt_subspace',
                            order=['FIM', 'Hybrid_05', 'Hybrid_1', 'Hybrid_2',
                                   'Hybrid_5', 'BFGS', 'SR1'],
                            hue_order=['fides 2D', 'fides full', 'ls_trf'])
            plt.tight_layout()
            plt.savefig(os.path.join(
                'evaluation',
                f'{MODEL_NAME}_{metric}_{EVALUATION_TYPE}.pdf'
            ))

    if EVALUATION_TYPE == 'adjoint':
        opt0 = 'ipopt'
        opt1 = 'fides.subspace=full.hessian=BFGS'
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
            'evaluation', f'{MODEL_NAME}_{EVALUATION_TYPE}.pdf'))