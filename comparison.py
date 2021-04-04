import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

from evaluate import (
    load_results, get_num_converged_per_grad, get_num_converged,
    ALGO_COLORS, ANALYSIS_ALGOS
)
from compile_petab import load_problem
from benchmark import set_solver_model_options

pd.set_option('max_columns', None)
pd.set_option('max_rows', None)


new_rc_params = {
    "font.family": 'Helvetica',
    "pdf.fonttype": 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
}
mpl.rcParams.update(new_rc_params)


def get_unique_starts_at_boundary(pars, lb, ub):
    return len(
        np.unique([
            str(sorted(np.where(np.isclose(par, lb, atol=1e-2, rtol=0.0)
                                | np.isclose(par, ub, atol=1e-2, rtol=0.0))))
            for par in pars
            if par is not None
        ])
    )


def get_number_boundary_optima(pars, iters, grads, lb, ub):
    return sum([
        n_iter > 0 and (
            np.isclose(par, lb, atol=1e-2, rtol=0.0).any() or
            np.isclose(par, ub, atol=1e-2, rtol=0.0).any()
        ) and
        np.linalg.norm(grad[np.where(
            np.isclose(par, lb, atol=1e-2, rtol=0.0) |
            np.isclose(par, ub, atol=1e-2, rtol=0.0)
        )]) ** 2 / np.linalg.norm(grad) ** 2 > 0.9
        for par, n_iter, grad in zip(pars, iters, grads)
        if par is not None and grad is not None
    ])


for analysis, algos in ANALYSIS_ALGOS.items():

    all_results = []

    models = ['Fujita_SciSignal2010', 'Boehm_JProteomeRes2014',
              'Fiedler_BMC2016', 'Crauste_CellSystems2017',
              'Brannmark_JBC2010', 'Weber_BMC2015', 'Zheng_PNAS2012']

    for model in models:

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
            if optimizer not in results:
                continue
            result = results[optimizer]
            all_results.append(
                {
                    'model': model,
                    'optimizer': optimizer,
                    'conv_count': get_num_converged(
                        result.optimize_result.get_for_key('fval'),
                        fmin
                    ),
                    'conv_per_grad': get_num_converged_per_grad(
                        result.optimize_result.get_for_key('fval'),
                        result.optimize_result.get_for_key('n_grad'),
                        fmin
                    ),
                    'unique_at_boundary': get_unique_starts_at_boundary(
                        result.optimize_result.get_for_key('x'),
                        problem.lb_full, problem.ub_full
                    ),
                    'boundary_minima': get_number_boundary_optima(
                        result.optimize_result.get_for_key('x'),
                        result.optimize_result.get_for_key('n_grad'),
                        result.optimize_result.get_for_key('grad'),
                        problem.lb_full, problem.ub_full
                    ),
                }
            )

    results = pd.DataFrame(all_results)

    if analysis == 'matlab':
        palette = [
            ALGO_COLORS.get(algo, ALGO_COLORS.get('ipopt'))
            for algo in algos
        ]
    elif analysis == 'curv':
        palette = 'tab20'
    elif analysis == 'hybrid':
        palette = 'Blues'
    elif analysis == 'stepback':
        palette = 'Set2'

    print(results)

    for metric in ['conv_count', 'conv_per_grad', 'unique_at_boundary',
                   'boundary_minima']:
        plt.subplots()
        g = sns.barplot(
            data=results,
            x='model',
            y=metric,
            hue='optimizer',
            hue_order=algos,
            palette=palette,
            bottom=1e-6 if metric == 'conv_per_grad' else 0.5,
        )
        g.set_xticklabels(g.get_xticklabels(), rotation=45)
        if metric in ['conv_count', 'conv_per_grad']:
            g.set_yscale('log')

        plt.tight_layout()
        plt.savefig(os.path.join(
            'evaluation', f'comparison_{analysis}_{metric}.pdf'
        ))
