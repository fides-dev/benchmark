import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

from evaluate import (
    load_results, get_num_converged_per_grad, get_num_converged,
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

    for model in ['Brannmark_JBC2010','Boehm_JProteomeRes2014',
                  'Fiedler_BMC2016', 'Crauste_CellSystems2017',
                  'Weber_BMC2015', 'Zheng_PNAS2012', 'Fujita_SciSignal2010',
                  'Beer_MolBioSystems2014']:

        hass_alias = {
            'Crauste_CellSystems2017': 'Crauste_ImmuneCells_CellSystems2017',
            'Beer_MolBioSystems2014': 'Beer_MolBiosyst2014',
        }

        petab_problem, problem = load_problem(model)
        set_solver_model_options(problem.objective.amici_solver,
                                 problem.objective.amici_model)

        results = {}
        for optimizer in algos:
            try:
                results[optimizer] = load_results(model, optimizer, '1000')
            except FileNotFoundError:
                pass

        fmin = np.min([
            result.optimize_result.list[0].fval
            for result in results.values()
        ])

        for optimizer in algos:
            result = results[optimizer]
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
