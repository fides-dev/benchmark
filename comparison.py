import os
import re
import petab
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import sklearn

from evaluate import (
    load_results_from_hdf5, get_num_converged_per_grad, get_num_converged,
    CONVERGENCE_THRESHOLD, ALGO_COLORS
)
from compile_petab import load_problem
from benchmark import set_solver_model_options


new_rc_params = {
    "font.family": 'Helvetica',
    "pdf.fonttype": 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
}
mpl.rcParams.update(new_rc_params)


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


for analysis, algos in {
    'matlab': ['fides.subspace=2D', 'fides.subspace=2D.hessian=Hybrid_2',
               'fmincon', 'lsqnonlin', 'ls_trf'],
    'curv': ['fides.subspace=2D',
             'fides.subspace=full',
             'fides.subspace=2D.hessian=SR1',
             'fides.subspace=full.hessian=SR1'],
    'hybrid': ['fides.subspace=2D', 'fides.subspace=2D.hessian=Hybrid_05',
               'fides.subspace=2D.hessian=Hybrid_1',
               'fides.subspace=2D.hessian=Hybrid_2',
               'fides.subspace=2D.hessian=Hybrid_5',
               'fides.subspace=2D.hessian=BFGS'],
}.items():

    all_results = []

    for model in ['Boehm_JProteomeRes2014', 'Fiedler_BMC2016',
                  'Brannmark_JBC2010', 'Crauste_CellSystems2017',
                  'Weber_BMC2015', 'Zheng_PNAS2012', 'Fujita_SciSignal2010',
                  'Beer_MolBioSystems2014']:

        hass_alias = {
            'Crauste_CellSystems2017': 'Crauste_ImmuneCells_CellSystems2017',
            'Beer_MolBioSystems2014': 'Beer_MolBiosyst2014',
        }

        hass_2019 = pd.read_excel(os.path.join(
            'Hass2019', f'{model}.xlsx'
        ), sheet_name='General Info')

        hass_2019_pars = pd.read_excel(os.path.join(
            'Hass2019', f'{model}.xlsx'
        ), sheet_name='Parameters')
        hass_2019_pars.parameter = hass_2019_pars.parameter.apply(
            lambda x: re.sub(r'log10\(([\w_]+)\)', r'\1', x)
        )

        petab_problem, problem = load_problem(model)
        set_solver_model_options(problem.objective.amici_solver,
                                 problem.objective.amici_model)

        if model == 'Weber_BMC2015':
            hass_2019_pars = hass_2019_pars.append(
                petab_problem.parameter_df.reset_index().loc[
                    [39, 38, 35, 36, 37],
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

        fmin = hass_2019.iloc[5, 1] / 2
        if model == 'Crauste_CellSystems2017':
            fmin = 190.96521897435176
        if model == 'Fiedler_BMC2016':
            # benchmark results worse than reference
            fmin = -56.86964545865438

        matlab_alias = {
            'fmincon': 'trust',
            'lsqnonlin': 'lsq'
        }

        for matlab_algo, alias in matlab_alias.items():
            if matlab_algo not in algos:
                continue
            hass2019_chis = np.genfromtxt(os.path.join(
                'Hass2019',
                f'{hass_alias.get(model, model)}_{alias}_chi2s.csv',
            ), delimiter=',')
            hass2019_iter = np.genfromtxt(os.path.join(
                'Hass2019',
                f'{hass_alias.get(model, model)}_{alias}_iter.csv',
            ), delimiter=',')
            hass2019_ps = np.genfromtxt(os.path.join(
                'Hass2019',
                f'{hass_alias.get(model, model)}_{alias}_ps.csv',
            ), delimiter=',')

            fvals_file = os.path.join(
                'Hass2019',
                f'{hass_alias.get(model, model)}_{alias}_fvals.csv',
            )
            if os.path.exists(fvals_file):
                hass2019_fvals = np.genfromtxt(fvals_file, delimiter=',')
            else:
                if model == 'Crauste_CellSystems2017':
                    chi2min = 19.6659294289557
                    hass2019_fvals = (hass2019_chis - chi2min)/2 + fmin
                else:
                    hass2019_fvals = np.array([
                        problem.objective(p[par_idx])
                        if not np.isnan(p).any() else np.NaN
                        for p in hass2019_ps
                    ])
                np.savetxt(
                    fvals_file, hass2019_fvals, delimiter=','
                )

            all_results.append(
                {
                    'model': model,
                    'optimizer': matlab_algo,
                    'n_converged': get_num_converged(
                        hass2019_fvals, fmin
                    ),
                    'conv_per_grad': get_num_converged_per_grad(
                        hass2019_fvals, hass2019_iter, fmin
                    ),
                    'dist': get_dist(
                        hass2019_fvals, fmin
                    )
                }
            )

        for optimizer in algos:
            if optimizer in matlab_alias:
                continue
            try:
                result = load_results_from_hdf5(model, optimizer, '1000')
                all_results.append(
                    {
                        'model': model,
                        'optimizer': optimizer,
                        'n_converged': get_num_converged(
                            result.optimize_result.get_for_key('fval'),
                            fmin
                        ),
                        'conv_per_grad': get_num_converged_per_grad(
                            result.optimize_result.get_for_key('fval'),
                            result.optimize_result.get_for_key('n_grad'),
                            fmin
                        ),
                        'dist': get_dist(
                            result.optimize_result.get_for_key('fval'),
                            fmin
                        )
                    }
                )
            except FileNotFoundError:
                pass

    results = pd.DataFrame(all_results)

    g = sns.FacetGrid(
        pd.melt(results, id_vars=['model', 'optimizer']),
        row='variable',
        sharey=False,
        legend_out=True
    )
    if analysis == 'matlab':
        palette = [
            ALGO_COLORS.get(algo, ALGO_COLORS.get('ipopt'))
            for algo in algos
        ]
    elif analysis == 'curv':
        palette = 'tab20'
    elif analysis == 'hybrid':
        palette = 'Blues'

    ax = g.map_dataframe(sns.barplot,
                         x='model',
                         y='value',
                         hue='optimizer',
                         hue_order=algos,
                         palette=palette)
    [plt.setp(ax.get_xticklabels(), rotation=90) for ax in g.axes.flat]
    [ax.set_yscale('log') for ax in g.axes.flat]
    g.add_legend()

    plt.tight_layout()
    plt.savefig(os.path.join(
        'evaluation', f'comparison_{analysis}.pdf'
    ))
