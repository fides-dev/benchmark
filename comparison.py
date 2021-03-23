import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

from evaluate import (
    load_results_from_hdf5, get_num_converged_per_grad, get_num_converged,
    get_dist, ALGO_COLORS
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

        petab_problem, problem = load_problem(model)
        set_solver_model_options(problem.objective.amici_solver,
                                 problem.objective.amici_model)

        matlab_alias = {
            'fmincon': 'trust',
            'lsqnonlin': 'lsq'
        }

        for matlab_algo, alias in matlab_alias.items():
            if matlab_algo not in algos:
                continue


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
