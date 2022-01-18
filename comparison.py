import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pypesto

from evaluate import (
    load_results, get_num_converged_per_grad, get_num_converged,
    ANALYSIS_ALGOS, ALGO_PALETTES, CONVERGENCE_THRESHOLDS, OPTIMIZER_FORWARD
)
from compile_petab import load_problem
from benchmark import set_solver_model_options

MODELS = [
    'Bachmann_MSB2011', 'Beer_MolBioSystems2014', 'Boehm_JProteomeRes2014',
    'Brannmark_JBC2010', 'Bruno_JExpBot2016',  'Crauste_CellSystems2017',
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

            results.loc[mrows, 'conv_count'] = results.loc[mrows, :].apply(
                lambda x:
                get_num_converged(x.fvals,
                                  fmin_model if not has_ebounds(x.optimizer)
                                  else min(x.fmin, fmin_model),
                                  threshold),
                axis=1
            )
            results.loc[mrows, 'conv_rate'] = results.loc[mrows, :].apply(
                lambda x:
                get_num_converged_per_grad(x.fvals, x.iter,
                                           fmin_model if not has_ebounds(
                                               x.optimizer)
                                           else min(x.fmin, fmin_model),
                                           threshold), axis=1
            )

            # compute improvement compared to ref algo
            ref_algo = 'fides.subspace=2D'
            if np.any(mrows & (results.optimizer == ref_algo)):
                ref_val = results.loc[
                    mrows & (results.optimizer == ref_algo), 'conv_rate'
                ].values[0]
                results.loc[mrows, 'improvement'] = \
                    results.loc[mrows, 'conv_rate'] / ref_val

                # sort fvals according to start id
                fvals = {
                    opt: np.asarray(results.loc[
                        mrows & (results.optimizer == opt),
                        'fvals'
                    ].values[0])[np.argsort([
                        int(start_id) for start_id in
                        results.loc[mrows & (results.optimizer == opt),
                                    'ids'].values[0]
                        if start_id is not None
                    ])]
                    for opt in results[mrows].optimizer.unique()
                }
                corr_max = 500
                for opt in results[mrows].optimizer.unique():
                    if len(fvals[opt]) < corr_max:
                        continue
                    if len(fvals[ref_algo]) < corr_max:
                        continue
                    results.loc[mrows & (results.optimizer == opt),
                                'fcorr'] = np.corrcoef(
                        np.log10(fvals[opt][:corr_max] + fmin_model),
                        np.log10(fvals[ref_algo][:corr_max] + fmin_model)
                    )[0, 1]

        if 'improvement' in results:
            for optimizer in results.optimizer.unique():
                results.loc[results.optimizer == optimizer,
                            'average improvement'] = \
                    10 ** results.loc[results.optimizer == optimizer,
                                      'improvement'].apply(np.log10).mean()

        if 'fcorr' in results:
            for optimizer in results.optimizer.unique():
                sel = results.optimizer == optimizer
                results.loc[sel, 'average fcorr'] = results.loc[sel,
                                                                'fcorr'].mean()

        results.drop(columns=['fvals', 'iter']).to_csv(
            os.path.join('evaluation', f'comparison_{threshold}.csv')
        )

        df = pd.melt(results, id_vars=['model', 'optimizer'],
                     value_vars=['unique_at_boundary', 'boundary_minima',
                                 'conv_count'])

        for analysis, algos in ANALYSIS_ALGOS.items():
            df_analysis = df[df.optimizer.isin(algos)]
            results_analysis = results[results.optimizer.isin(algos)]
            palette = ALGO_PALETTES[analysis]

            # conv counts plot
            g = sns.FacetGrid(
                df_analysis,  row='variable',
                sharex=True, sharey=True,
                height=3, aspect=2,
            )
            g.map_dataframe(sns.barplot, x='model', y='value',
                            hue='optimizer', hue_order=algos,
                            palette=palette, bottom=1e0)

            g.set(yscale='log', ylim=(1e0, 1e3))
            for ax in g.axes.ravel():
                ax.set_xticklabels(ax.get_xticklabels(),
                                   rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(
                'evaluation', f'comparison_{analysis}_{threshold}_counts.pdf'
            ))

            # conv rate plot
            plt.figure(figsize=(9, 5))
            g = sns.barplot(
                data=results_analysis,
                x='model', y='conv_rate', hue='optimizer', hue_order=algos,
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

            # iter plot
            df_iter = pd.DataFrame(
                [(d, tup.model, tup.optimizer)
                 for tup in results_analysis.itertuples()
                 for d in tup.iter],
                columns=['iter', 'model', 'optimizer']
            )
            df_iter.iter = df_iter.iter.apply(np.log10)
            plt.figure(figsize=(9, 5))
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

            # corr plot
            plt.figure(figsize=(9, 5))
            g = sns.barplot(
                data=results_analysis, hue_order=algos, palette=palette,
                x='model', hue='optimizer', y='fcorr'
            )
            g.set_xticklabels(g.get_xticklabels(), rotation=45, ha='right')
            g.set(yscale='linear', ylim=[-0.25, 1])
            plt.tight_layout()
            plt.savefig(os.path.join(
                'evaluation', f'comparison_{analysis}_fcorr.pdf'
            ))
