import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pypesto
import re

from scipy.stats import pearsonr

from evaluate import (
    load_results, get_num_converged_per_grad, get_num_converged,
    ANALYSIS_ALGOS, ALGO_PALETTES, ALGO_COLORS, CONVERGENCE_THRESHOLDS,
    OPTIMIZER_FORWARD
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

        ref_algos = {
            'GN': 'fides.subspace=2D'
            if threshold != CONVERGENCE_THRESHOLDS[0]
            else 'fides.subspace=2D.hessian=FIMe',
            'BFGS': 'fides.subspace=2D.hessian=BFGS',
            'SR1': 'fides.subspace=2D.hessian=SR1',
        }

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
            results.loc[mrows, 'perf'] = results.loc[mrows, :].apply(
                lambda x:
                get_num_converged_per_grad(x.fvals, x.iter,
                                           fmin_model if not has_ebounds(
                                               x.optimizer)
                                           else min(x.fmin, fmin_model),
                                           threshold), axis=1
            )

            results.loc[mrows, 'best performer'] = \
                results.loc[mrows, 'perf'] > \
                results.loc[mrows, 'perf'].max() * 0.85

            results['conv rate'] = results.iter.apply(
                lambda x: 1 / np.sum(x) if np.sum(x) > 0 else np.nan
            )
            perfs = ['perf', 'conv rate', 'conv count']

            # compute improvement compared to ref algo
            improvements = []
            for ref_name, ref_algo in ref_algos.items():
                if np.any(mrows & (results.optimizer == ref_algo)):
                    for perf in perfs:
                        ref_val = results.loc[
                            mrows & (results.optimizer == ref_algo), perf
                        ].values[0]
                        val = results.loc[mrows, perf] / ref_val
                        results.loc[mrows, f'improvement ({ref_name}) {perf}']\
                            = val

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

                    ref_conv_ids = set(
                        opt_ids[ref_algo][:conv_counts[ref_algo]]
                    )
                    for opt in results[mrows].optimizer.unique():

                        opt_conv_ids = set(opt_ids[opt][:conv_counts[opt]])

                        n_min = min(len(opt_conv_ids), len(ref_conv_ids))
                        results.loc[mrows & (results.optimizer == opt),
                                    f'overlap ({ref_name})'] = len(
                            opt_conv_ids.intersection(ref_conv_ids)
                        ) / n_min if n_min > 0 else 0.0

                else:
                    print(f'No results for {ref_algo} for {model}')

        for ref in ref_algos.keys():
            for perf in perfs:
                if f'improvement ({ref}) {perf}' in results:
                    for optimizer in results.optimizer.unique():
                        val = 10 ** results.loc[
                            results.optimizer == optimizer,
                            f'improvement ({ref}) {perf}'
                        ].apply(np.log10).mean()
                        results.loc[results.optimizer == optimizer,
                                    f'average improvement ({ref}) {perf}'] =\
                            val

            if f'overlap ({ref})' in results:
                for optimizer in results.optimizer.unique():
                    sel = results.optimizer == optimizer
                    results.loc[sel, f'average overlap ({ref})'] = \
                        results.loc[sel, f'overlap ({ref})'].mean()

        results.drop(columns=['fvals', 'iter', 'ids']).to_csv(
            os.path.join('evaluation', f'comparison_{threshold}.csv')
        )

        df = pd.melt(results, id_vars=['model', 'optimizer'],
                     value_vars=['unique_at_boundary', 'boundary_minima',
                                 'conv count'])

        for analysis, algos in ANALYSIS_ALGOS.items():
            if threshold == CONVERGENCE_THRESHOLDS[0]:
                algos = [a if a != 'fides.subspace=2D' else
                         'fides.subspace=2D.hessian=FIMe'
                         for a in algos if a != 'random']
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

            overlaps = [f'overlap ({ref})' for ref in ref_algos.keys()]

            for ref_name, ref_algo in ref_algos.items():
                if analysis not in ['curv', 'hybridB'] and ref_name != 'GN':
                    continue
                if analysis == 'hybridB' and ref_name == 'SR1':
                    continue
                for model in MODELS:
                    mrows = results_analysis.model == model.split('_')[0]
                    for stat in stat_columns:
                        ref_val = results_analysis.loc[
                            mrows & (results_analysis.optimizer == ref_algo),
                            stat
                        ].values[0]
                        results_analysis.loc[mrows,
                                             f'change ({ref_name}) {stat}'] = \
                            results_analysis.loc[mrows, stat] - ref_val

            palette = ALGO_PALETTES[analysis]

            for perf in perfs:
                for opt in results_analysis.optimizer.unique():
                    if opt == ref_algos['GN']:
                        continue
                    results_opt = results_analysis.loc[
                        results_analysis.optimizer == opt, :
                    ]
                    better_than_ref = results_opt[
                        results_opt[f'improvement (GN) {perf}'] > 1.15
                    ]
                    similar_to_ref = results_opt[
                        (results_opt[f'improvement (GN) {perf}'] < 1.15) &
                        (results_opt[f'improvement (GN) {perf}'] > 0.85)
                    ]
                    worse_than_ref = results_opt[
                        (results_opt[f'improvement (GN) {perf}'] < 0.85) &
                        (results_opt[f'improvement (GN) {perf}'] > 0.0)
                    ]
                    failed = results_opt[
                        (results_opt[f'improvement (GN) {perf}'] < 0.01)
                    ]
                    worked = results_opt[
                        (results_opt[f'improvement (GN) {perf}'] > 0.01)
                    ]
                    best_performer = results_opt[results_opt['best performer']]

                    evaluations = [
                        (f'had better {perf}', better_than_ref),
                        (f'had similar {perf}', similar_to_ref),
                        (f'had worse {perf}', worse_than_ref),
                    ]
                    if perf == 'perf':
                        evaluations += [
                            ('worked', worked),
                            ('failed', failed),
                            ('was best performer', best_performer)
                        ]

                    for predicate, frame in evaluations:
                        models = ", ".join([
                            f"\\textit{{{model}}}" for model in frame.model
                        ])
                        if frame.empty:
                            continue

                        print(
                            f'{opt} {predicate} on {len(frame)} problems '
                            f'({min(frame["improvement (GN) perf"]):.2f} '
                            f'to {max(frame["improvement (GN) perf"]):.2f} '
                            f'fold change; '
                            f'{frame["improvement (GN) perf"].values[0]:.2f} '
                            f'average; {models})'
                        )

            # conv counts plot
            g = sns.FacetGrid(
                data=df_analysis, row='variable',
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

            # performance plot
            plt.figure(figsize=(9, 4))
            g = sns.barplot(
                data=results_analysis,
                x='model', y='perf', hue='optimizer', hue_order=algos,
                palette=palette,
                bottom=1e-7,
            )
            g.set_xticklabels(g.get_xticklabels(), rotation=45, ha='right')
            g.set(yscale='log', ylim=(1e-7, 1e-1))
            plt.tight_layout()
            plt.savefig(os.path.join(
                'evaluation',
                f'comparison_{analysis}_{threshold}_perf.pdf'
            ))

            # similarity plot
            plt.figure(figsize=(9, 4))
            g = sns.barplot(
                data=results_analysis, hue_order=algos, palette=palette,
                x='model', hue='optimizer', y='overlap (GN)'
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

            stat_changes = [
                f'change ({ref}) {stat}'
                for stat in stat_columns
                for ref in ref_algos.keys()
            ]
            stat_changes = [change for change in stat_changes
                            if change in results_analysis.columns]

            # decompose perf change into convergence count + iteration
            group_vars = ['model', 'optimizer']
            improvements_perfs = [
                f'improvement ({ref}) {perf}' for perf in perfs
                for ref in ref_algos.keys()
            ]
            improvements_perfs = [imp for imp in improvements_perfs
                                  if imp in results_analysis.columns]

            for imp in improvements_perfs:
                results_analysis[imp] = \
                    results_analysis[imp].apply(np.log10)
                results_analysis.loc[results_analysis[imp] < -2, imp] = -2
                results_analysis.loc[results_analysis[imp] > 2, imp] = 2

            df_improvement = pd.melt(
                results_analysis,
                id_vars=group_vars,
                value_vars=improvements_perfs,
                var_name='improvement var',
                value_name='improvement'
            )

            g = sns.FacetGrid(
                data=df_improvement, row='improvement var',
                row_order=[
                    f'improvement (GN) {perf}' for perf in perfs
                ],
                height=2, aspect=3,
            )
            g.axes[0, 0].axhline(np.log10(1.15), color='k', linestyle='--')
            g.axes[0, 0].axhline(np.log10(0.85), color='k', linestyle='--')
            g.map_dataframe(
                sns.barplot, x='model', y='improvement',
                hue='optimizer', hue_order=algos, palette=palette,
            )
            g.set(ylim=[-2, 2])
            g.axes[-1, 0].set_xticklabels(g.axes[-1, 0].get_xticklabels(),
                                          rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(
                'evaluation',
                f'comparison_{analysis}_{threshold}_improvement_decomp.pdf'
            ))

            for perf in perfs:
                results_analysis[perf] = results_analysis[perf].apply(np.log10)
                results_analysis.loc[results_analysis[perf] < -7, perf] = -7

            for col_type, cols in zip(['stat', 'change', 'overlap'],
                                      [stat_columns, stat_changes, overlaps]):
                if not cols:
                    continue
                for opt in results_analysis.optimizer.unique():
                    data = results_analysis[results_analysis.optimizer == opt]
                    if threshold == CONVERGENCE_THRESHOLDS[0]:
                        if opt == 'fides.subspace=2D.hessian=FIMe':
                            # only changes color & skipping
                            opt = 'fides.subspace=2D'
                    for x in cols:
                        for y in perfs + improvements_perfs:
                            points = data.dropna(subset=[x, y])
                            if points.empty:
                                continue
                            r, p = pearsonr(points[x], points[y])

                            if np.isnan(p) or p > 0.05:
                                continue

                            if analysis == 'curv':
                                if (x.startswith('change (GN)') or
                                    y.startswith('improvement (GN)')) and (
                                    opt not in [
                                        'fides.subspace=full',
                                        'fides.subspace=2D.hessian=BFGS',
                                        'fides.subspace=2D.hessian=SR1',
                                    ]
                                ):
                                    continue

                                if (x.startswith('change (BFGS)') or
                                    y.startswith('improvement (BFGS)')) and (
                                    opt not in [
                                        'fides.subspace=full.hessian=BFGS',
                                        'fides.subspace=2D.hessian=SR1',
                                    ]
                                ):
                                    continue

                                if (x.startswith('change (SR1)') or
                                    y.startswith('improvement (SR1)')) and (
                                    opt not in [
                                        'fides.subspace=full.hessian=SR1',
                                        'fides.subspace=2D.hessian=BFGS',
                                    ]
                                ):
                                    continue

                            if analysis in ['curv', 'hybridB']:
                                my = re.match(r'improvement \(([\w]+)\)', y)
                                mx = re.match(r'change \(([\w]+)\)', x)

                                if mx and my and mx.group(1) != my.group(1):
                                    continue

                            plt.figure(figsize=(4, 4))
                            g = sns.regplot(
                                data=points, x=x, y=y,
                                color=ALGO_COLORS[analysis][opt]
                            )
                            g.text(.05, .8, f'r={r:.2f}, p={p:.2e}',
                                   transform=g.transAxes)
                            plt.tight_layout()
                            plt.savefig(os.path.join(
                                'evaluation',
                                f'comparison_{analysis}_{threshold}_'
                                f'{opt}_{x}_vs_{y}.pdf'
                            ))

                df_plot = pd.melt(
                    results_analysis,
                    id_vars=group_vars + perfs + improvements_perfs,
                    value_vars=cols,
                    var_name=f'{col_type} var',
                    value_name=col_type
                )
                if col_type == 'change':
                    if analysis == 'curv':
                        df_plot = df_plot[
                            df_plot[f'{col_type} var'].apply(
                                lambda x: x.startswith('change (GN)')
                            ) & (
                                (df_plot.optimizer ==
                                 'fides.subspace=2D.hessian=BFGS') |
                                (df_plot.optimizer ==
                                 'fides.subspace=2D.hessian=SR1')
                            )
                        ]
                    else:
                        df_plot = df_plot[
                            df_plot.optimizer != ref_algos['GN']
                        ]
                else:
                    data = results_analysis
                data = pd.melt(
                    df_plot,
                    value_vars=perfs + improvements_perfs,
                    id_vars=[f'{col_type} var', col_type] + group_vars,
                ).dropna(subset=['value', col_type])
                if data.empty:
                    continue
                g = sns.lmplot(
                    data=data,
                    sharex=False, sharey=False,
                    row=f'{col_type} var', col='variable',
                    x=col_type, y='value',
                    hue='optimizer', hue_order=algos, palette=palette,
                )
                plt.tight_layout()
                plt.savefig(os.path.join(
                    'evaluation',
                    f'comparison_{analysis}_{threshold}_'
                    f'{col_type}_vs_perf.pdf'
                ))

