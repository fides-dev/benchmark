import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pypesto

from evaluate import (
    load_results, get_num_converged_per_grad, get_num_converged,
    ALGO_COLORS, ANALYSIS_ALGOS
)
from compile_petab import load_problem
from benchmark import set_solver_model_options

MODELS = [
    'Bachmann_MSB2011', 'Beer_MolBioSystems2014',
    'Boehm_JProteomeRes2014',
    'Brannmark_JBC2010', 'Bruno_JExpBot2016', 'Crauste_CellSystems2017',
    'Fiedler_BMC2016', 'Fujita_SciSignal2010',
    #'Isensee_JCB2018',
    'Lucarelli_CellSystems2018', 'Schwen_PONE2014', 'Weber_BMC2015',
    'Zheng_PNAS2012'
]


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


if __name__ == '__main__':

    pd.set_option('max_columns', None)
    pd.set_option('max_rows', None)

    new_rc_params = {
        "font.family": 'Helvetica',
        "pdf.fonttype": 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
    }
    mpl.rcParams.update(new_rc_params)

    for analysis, algos in ANALYSIS_ALGOS.items():

        all_results = []

        for model in MODELS:
            petab_problem, problem = load_problem(model)
            if isinstance(problem.objective, pypesto.AmiciObjective):
                objective = problem.objective
            else:
                objective = problem.objective._objectives[0]
            set_solver_model_options(objective.amici_solver,
                                     objective.amici_model)

            results = {}
            for optimizer in algos:
                try:
                    results[optimizer] = load_results(model, optimizer, '1000')
                except (FileNotFoundError, IOError) as err:
                    print(f'Failed loading: {err}')

            fmin_all = np.nanmin([
                result.optimize_result.list[0].fval
                for optimizer, result in results.items()
                if 'ebounds=True' not in optimizer.split('.')
            ])

            for optimizer in algos:
                if optimizer not in results:
                    continue
                result = results[optimizer]

                if 'ebounds=True' in optimizer.split('.'):
                    ubs = np.asarray([
                        ub + 1 if scale == 'log10'
                        else ub*10
                        for ub, scale in zip(
                            problem.ub_full, problem.x_scales
                        )
                    ])
                    lbs = np.asarray([
                        lb + 1 if scale == 'log10'
                        else lb*10
                        for lb, scale in zip(
                            problem.lb_full, problem.x_scales
                        )
                    ])
                    fmin = np.min([
                        results[optimizer].optimize_result.list[0].fval,
                        fmin_all
                    ])
                else:
                    ubs = problem.ub_full
                    lbs = problem.lb_full
                    fmin = fmin_all

                n_iter = np.asarray(
                    result.optimize_result.get_for_key('n_grad')
                ) + np.asarray(
                    result.optimize_result.get_for_key('n_sres')
                )

                all_results.append({
                    'model': model.split('_')[0],
                    'optimizer': optimizer,
                    'iter': n_iter,
                    'conv_count': get_num_converged(
                        result.optimize_result.get_for_key('fval'),
                        fmin
                    ),
                    'conv_per_grad': get_num_converged_per_grad(
                        result.optimize_result.get_for_key('fval'), n_iter,
                        fmin
                    ),
                    'unique_at_boundary': get_unique_starts_at_boundary(
                        result.optimize_result.get_for_key('x'),
                        lbs, ubs
                    ),
                    'boundary_minima': get_number_boundary_optima(
                        result.optimize_result.get_for_key('x'),
                        n_iter,
                        result.optimize_result.get_for_key('grad'),
                        lbs, ubs
                    ),
                })

        results = pd.DataFrame(all_results)

        if analysis == 'matlab':
            palette = [
                ALGO_COLORS.get(algo, ALGO_COLORS.get('ipopt'))
                for algo in algos
            ]
        elif analysis == 'curv':
            palette = 'tab20'
        elif analysis == 'hybrid':
            palette = 'Dark2'
        elif analysis in ['hybridB', 'hybridS', 'hybridB0', 'hybridS0']:
            palette = 'Blues'
        elif analysis == 'stepback':
            palette = 'Set2'
        for model in MODELS:
            model = model.split('_')[0]
            results.loc[results.model == model, 'improvement'] = \
                results.loc[results.model == model, 'conv_per_grad'] / \
                results.loc[(results.model == model) &
                            (results.optimizer == 'fides.subspace=2D'),
                            'conv_per_grad'].values[0]

        for optimizer in results.optimizer.unique():
            if 'improvement' in results:
                results.loc[results.optimizer == optimizer,
                            'average improvement'] = \
                    10 ** results.loc[results.optimizer == optimizer,
                                      'improvement'].apply(np.log10).mean()

        print(results)

        results.to_csv(os.path.join('evaluation', f'comparison_{analysis}.csv'))

        for metric in ['conv_count', 'conv_per_grad', 'unique_at_boundary',
                       'boundary_minima']:

            bottoms = {
                'conv_per_grad': 5e-7,
                'conv_count':  0,
                'unique_at_boundary': 0,
                'boundary_minima': 0,
            }

            tops = {
                'conv_per_grad': 1e-1,
                'conv_count': 1e3,
                'unique_at_boundary': 1e3,
                'boundary_minima': 3e2,
            }

            plt.subplots()
            g = sns.barplot(
                data=results,
                x='model',
                y=metric,
                hue='optimizer',
                hue_order=algos,
                palette=palette,
                bottom=bottoms[metric],
            )
            g.set_xticklabels(g.get_xticklabels(), rotation=45, ha='right')
            if metric in ['conv_per_grad']:
                g.set_yscale('log')
            g.set(ylim=(bottoms[metric], tops[metric]))

            plt.tight_layout()
            plt.savefig(os.path.join(
                'evaluation', f'comparison_{analysis}_{metric}.pdf'
            ))

        df_iter = pd.DataFrame(
            [(d, tup.model, tup.optimizer)
             for tup in results.itertuples() for d in tup.iter],
            columns=['iter', 'model', 'optimizer']
        )
        df_iter.iter = df_iter.iter.apply(np.log10)
        plt.subplots()
        g = sns.boxplot(
            data=df_iter, hue_order=algos, palette=palette,
            x='model', hue='optimizer', y='iter',
        )
        g.set_xticklabels(g.get_xticklabels(), rotation=45, ha='right')
        g.set_ylim([0, 5])
        plt.tight_layout()
        plt.savefig(os.path.join(
            'evaluation', f'comparison_{analysis}_iter.pdf'
        ))
