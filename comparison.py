import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pypesto

from scipy.stats import skew

from evaluate import (
    load_results, get_num_converged_per_grad, get_num_converged,
    ANALYSIS_ALGOS, ALGO_PALETTES, CONVERGENCE_THRESHOLDS, OPTIMIZER_FORWARD
)
from compile_petab import load_problem
from benchmark import set_solver_model_options

MODELS = [
    'Bachmann_MSB2011', 'Beer_MolBioSystems2014', 'Boehm_JProteomeRes2014',
    'Brannmark_JBC2010', 'Bruno_JExpBot2016', 'Crauste_CellSystems2017',
    'Fiedler_BMC2016', 'Fujita_SciSignal2010', 'Isensee_JCB2018',
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


OPTIMIZERS = OPTIMIZER_FORWARD + ['fmincon', 'lsqnonlin']

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

    all_results = []

    hdf_key = 'results'
    hdf_file = os.path.join('evaluation', 'comparison.h5')

    if not os.path.exists(hdf_file):
        for model in MODELS:
            petab_problem, problem = load_problem(model)
            if isinstance(problem.objective, pypesto.AmiciObjective):
                objective = problem.objective
            else:
                objective = problem.objective._objectives[0]
            set_solver_model_options(objective.amici_solver,
                                     objective.amici_model)

            for optimizer in OPTIMIZERS:
                try:
                    result = load_results(model, optimizer, '1000')
                except (FileNotFoundError, IOError) as err:
                    print(f'Failed loading: {err}')
                    continue

                ebound_option = next((
                    option
                    for option in optimizer.split('.')
                    if option.startswith('ebounds=')
                ), None)

                if ebound_option is not None:
                    ebound = float(ebound_option.split('=')[1])
                    ubs = np.asarray([
                        ub + np.log10(ebound) if scale == 'log10'
                        else ub * ebound
                        for ub, scale in zip(
                            problem.ub_full, problem.x_scales
                        )
                    ])
                    lbs = np.asarray([
                        lb - np.log10(ebound) if scale == 'log10'
                        else lb * ebound
                        for lb, scale in zip(
                            problem.lb_full, problem.x_scales
                        )
                    ])
                else:
                    ubs = problem.ub_full
                    lbs = problem.lb_full

                n_iter = np.asarray(
                    result.optimize_result.get_for_key('n_grad')
                ) + np.asarray(
                    result.optimize_result.get_for_key('n_sres')
                )

                all_results.append({
                    'model': model.split('_')[0],
                    'optimizer': optimizer,
                    'iter': n_iter,
                    'ids': result.optimize_result.get_for_key('id'),
                    'fvals': result.optimize_result.get_for_key('fval'),
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
        results['fmin'] = results['fvals'].apply(np.nanmin)
        results.to_hdf(hdf_file, hdf_key)
    else:
        results = pd.read_hdf(hdf_file, hdf_key)

    for threshold in CONVERGENCE_THRESHOLDS:
        for model in MODELS:
            mrows = results.model == model.split('_')[0]

            def has_ebounds(optimizer):
                return any(
                    option.startswith('ebounds=')
                    for option in optimizer.split('.')
                )


            fmin_model = results.loc[
                mrows & np.logical_not(results.optimizer.apply(has_ebounds)),
                'fmin'
            ].min()

            results.loc[mrows, 'conv count'] = results.loc[mrows, :].apply(
                lambda x:
                get_num_converged(x.fvals,
                                  fmin_model if not has_ebounds(x.optimizer)
                                  else min(x.fmin, fmin_model),
                                  threshold),
                axis=1
            )
            results.loc[mrows, 'conv rate'] = results.loc[mrows, :].apply(
                lambda x:
                get_num_converged_per_grad(x.fvals, x.iter,
                                           fmin_model if not has_ebounds(
                                               x.optimizer)
                                           else min(x.fmin, fmin_model),
                                           threshold), axis=1
            )

            results['mean iter'] = results.iter.apply(np.mean)
            results['skew iter'] = results.iter.apply(lambda x: skew(
                np.log10(x))
            )

            # compute improvement compared to ref algo
            ref_algo = 'fides.subspace=2D'
            if np.any(mrows & (results.optimizer == ref_algo)):
                for metric in ['conv rate', 'conv count', 'mean iter',
                               'skew iter']:
                    ref_val = results.loc[
                        mrows & (results.optimizer == ref_algo), metric
                    ].values[0]
                    val = results.loc[mrows, metric] / ref_val
                    results.loc[mrows, f'improvement {metric}'] = \
                        1 / val if metric == 'mean iter' else val

                opt_ids = {
                    opt: results.loc[mrows & (results.optimizer == opt),
                                     'ids'].values[0]
                    for opt in results[mrows].optimizer.unique()
                }

                conv_counts = {
                    opt: int(results.loc[
                        mrows & (results.optimizer == opt), 'conv count'
                    ].values[0])
                    for opt in results[mrows].optimizer.unique()
                }

                ref_conv_ids = set(opt_ids[ref_algo][:conv_counts[ref_algo]])
                for opt in results[mrows].optimizer.unique():

                    opt_conv_ids = set(opt_ids[opt][:conv_counts[opt]])

                    n_min = min(len(opt_conv_ids), len(ref_conv_ids))
                    results.loc[mrows & (results.optimizer == opt),
                                'overlap'] = len(
                        opt_conv_ids.intersection(ref_conv_ids)
                    ) / n_min if n_min > 0 else 0.0

            else:
                print(f'No results for {ref_algo} for {model}')

        for metric in ['conv rate', 'conv count', 'mean iter']:
            if f'improvement {metric}' in results:
                for optimizer in results.optimizer.unique():
                    results.loc[results.optimizer == optimizer,
                                f'average improvement {metric}'] = \
                        10 ** results.loc[
                            results.optimizer == optimizer,
                            f'improvement {metric}'
                        ].apply(np.log10).mean()

        if 'overlap' in results:
            for optimizer in results.optimizer.unique():
                sel = results.optimizer == optimizer
                results.loc[sel, f'average overlap'] = \
                    results.loc[sel, 'overlap'].mean()

        results.drop(columns=['fvals', 'iter', 'ids']).to_csv(
            os.path.join('evaluation', f'comparison_{threshold}.csv')
        )

        df = pd.melt(results, id_vars=['model', 'optimizer'],
                     value_vars=['unique_at_boundary', 'boundary_minima',
                                 'conv count'])

        for analysis, algos in ANALYSIS_ALGOS.items():
            df_analysis = df[df.optimizer.isin(algos)]
            results_analysis = results[results.optimizer.isin(algos)].copy()
            if analysis != 'matlab':
                stats = pd.read_csv(os.path.join('evaluation',
                                                 f'stats_{analysis}.csv'))

                stat_columns = [
                    stat for stat in stats.columns
                    if stat not in ['model', 'optimizer', 'converged', 'iter']
                ]
                for _, row in stats.iterrows():
                    for stat in stat_columns:
                        results_analysis.loc[
                            (results_analysis.model == row.model) &
                            (results_analysis.optimizer == row.optimizer),
                            stat
                        ] = row[stat]
            else:
                stat_columns = []

            palette = ALGO_PALETTES[analysis]

            # conv counts plot
            g = sns.FacetGrid(
                df_analysis, row='variable',
                sharex=True, sharey=True,
                height=3, aspect=2,
            )
            g.map_dataframe(sns.barplot, x='model', y='value',
                            hue='optimizer', hue_order=algos,
                            palette=palette, bottom=1e0)

            g.set(yscale='log', ylim=(10**0.5, 10**3.5))
            for ax in g.axes.ravel():
                ax.set_xticklabels(ax.get_xticklabels(),
                                   rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(
                'evaluation', f'comparison_{analysis}_{threshold}_counts.pdf'
            ))

            # conv rate plot
            plt.figure(figsize=(9, 4))
            g = sns.barplot(
                data=results_analysis,
                x='model', y='conv rate', hue='optimizer', hue_order=algos,
                palette=palette,
                bottom=1e-7,
            )
            g.set_xticklabels(g.get_xticklabels(), rotation=45, ha='right')
            g.set(yscale='log', ylim=(1e-7, 1e-1))
            plt.tight_layout()
            plt.savefig(os.path.join(
                'evaluation',
                f'comparison_{analysis}_{threshold}_conv_rate.pdf'
            ))

            # similarity plot
            plt.figure(figsize=(9, 4))
            g = sns.barplot(
                data=results_analysis, hue_order=algos, palette=palette,
                x='model', hue='optimizer', y='overlap'
            )
            g.set_xticklabels(g.get_xticklabels(), rotation=45, ha='right')
            g.set(yscale='linear', ylim=[0, 1])
            plt.tight_layout()
            plt.savefig(os.path.join(
                'evaluation',
                f'comparison_{analysis}_{threshold}_overlap.pdf'
            ))

            # iter plot
            df_iter = pd.DataFrame(
                [(d, tup.model, tup.optimizer)
                 for tup in results_analysis.itertuples()
                 for d in tup.iter],
                columns=['iter', 'model', 'optimizer']
            )
            df_iter.iter = df_iter.iter.apply(np.log10)
            plt.figure(figsize=(9, 4))
            g = sns.boxplot(
                data=df_iter, hue_order=algos, palette=palette,
                x='model', hue='optimizer', y='iter'
            )
            g.set_xticklabels(g.get_xticklabels(), rotation=45, ha='right')
            g.set_ylim([-0.5, 5.5])
            plt.tight_layout()
            plt.savefig(os.path.join(
                'evaluation', f'comparison_{analysis}_iter.pdf'
            ))

            # improvement plot
            group_vars = ['model', 'optimizer']
            improvements = ['improvement conv count',
                            'improvement mean iter',
                            'improvement conv rate']
            df_improvement = pd.melt(
                results_analysis,
                id_vars=group_vars + stat_columns,
                value_vars=improvements,
                var_name='improvement var',
                value_name='improvement'
            )
            df_improvement.improvement.apply(np.log10)

            plt.figure(figsize=(9, 4))
            g = sns.FacetGrid(
                df_improvement, row='improvement var',
                row_order=improvements,
            )
            g.map_dataframe(
                sns.barplot, x='model', y='improvement',
                hue='improvement var', hue_order=algos, palette=palette,
                bottom=-2,
            )
            g.set(ylim=[-2, 2])
            plt.tight_layout()
            plt.savefig(os.path.join(
                'evaluation', f'comparison_{analysis}_improvements.pdf'
            ))

            # stats comparison
            df_improvement[df_improvement.improvement < -2].improvement = -2
            df_stats = pd.melt(
                df_improvement,
                id_vars=group_vars + ['improvement var', 'improvement'],
                value_vars=stat_columns,
                var_name='stat var',
                value_name='stat'
            )
            g = sns.lmplot(
                df_stats,
                sharex=False, sharey=True,
                row='stat var', column='improvement var',
                x='stat', y='improvement',
                hue='optimizer', hue_order=algos, palette=palette,
            )
            plt.tight_layout()
            plt.savefig(os.path.join(
                'evaluation', f'comparison_{analysis}_stats.pdf'
            ))
