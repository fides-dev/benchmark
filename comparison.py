import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

from evaluate import (
    load_results, get_num_converged_per_grad, ALGO_COLORS
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


for analysis, algos in {
    'matlab': [x for x in ALGO_COLORS if x != 'ipopt'],
    'curv': ['fides.subspace=2D',
             'fides.subspace=full',
             'fides.subspace=2D.hessian=SR1',
             'fides.subspace=full.hessian=SR1'],
    'hybrid': ['fides.subspace=2D', 'fides.subspace=2D.hessian=Hybrid_5',
               'fides.subspace=2D.hessian=Hybrid_2',
               'fides.subspace=2D.hessian=Hybrid_1',
               'fides.subspace=2D.hessian=Hybrid_05',
               'fides.subspace=2D.hessian=BFGS'],
}.items():

    all_results = []

    for model in ['Zheng_PNAS2012', 'Fiedler_BMC2016',
                  'Crauste_CellSystems2017',
                  'Brannmark_JBC2010', 'Weber_BMC2015',
                  'Boehm_JProteomeRes2014']:

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
                    'conv_per_grad': get_num_converged_per_grad(
                        result.optimize_result.get_for_key('fval'),
                        result.optimize_result.get_for_key('n_grad'),
                        fmin
                    ),
                    'unique_at_boundary': get_unique_starts_at_boundary(
                        result.optimize_result.get_for_key('x'),
                        problem.lb if optimizer in []
                        else problem.lb_full,
                        problem.ub if optimizer in []
                        else problem.ub_full
                    ),
                    'boundary_minima': get_number_boundary_optima(
                        result.optimize_result.get_for_key('x'),
                        result.optimize_result.get_for_key('n_grad'),
                        result.optimize_result.get_for_key('grad'),
                        problem.lb if optimizer in []
                        else problem.lb_full,
                        problem.ub if optimizer in []
                        else problem.ub_full
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

    plt.subplots()
    g = sns.barplot(
        data=results,
        x='model',
        y='conv_per_grad',
        hue='optimizer',
        hue_order=algos,
        palette=palette,
        bottom=1e-6,
    )
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    g.set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(
        'evaluation', f'comparison_{analysis}.pdf'
    ))

    plt.subplots()
    g = sns.barplot(
        data=results,
        x='model',
        y='unique_at_boundary',
        hue='optimizer',
        hue_order=algos,
        palette=palette,
    )
    g.set_xticklabels(g.get_xticklabels(), rotation=90)

    plt.tight_layout()
    plt.savefig(os.path.join(
        'evaluation', f'comparison_{analysis}_boundary.pdf'
    ))

    plt.subplots()
    g = sns.barplot(
        data=results,
        x='model',
        y='boundary_minima',
        hue='optimizer',
        hue_order=algos,
        palette=palette,
    )
    g.set_xticklabels(g.get_xticklabels(), rotation=90)

    plt.tight_layout()
    plt.savefig(os.path.join(
        'evaluation', f'comparison_{analysis}_boundary_minima.pdf'
    ))
