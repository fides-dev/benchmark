import sys
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import petab.visualize
import amici.petab_objective
import pypesto.petab
import petab

from pypesto.store import OptimizationResultHDF5Reader
from pypesto.visualize import waterfall, create_references
from pypesto.objective.history import CsvHistory
from matplotlib import cm
from compile_petab import folder_base


MODEL_NAME = sys.argv[1]
EVALUATION_TYPE = sys.argv[2]

hdf5_files = [r for r in os.listdir('results')
              if r.startswith(MODEL_NAME) and r.endswith('.hdf5')]

yaml_config = os.path.join(folder_base, MODEL_NAME, MODEL_NAME + '.yaml')
petab_problem = petab.Problem.from_yaml(yaml_config)
importer = pypesto.petab.PetabImporter(petab_problem)
problem = importer.create_problem()
model = importer.create_model()
solver = importer.create_solver()

solver.setMaxSteps(int(1e4))
solver.setAbsoluteTolerance(1e-8)
solver.setRelativeTolerance(1e-8)

if MODEL_NAME == 'Chen_MSB2009':
    solver.setMaxSteps(int(2e5))

all_results = []
for hdf_results_file in hdf5_files:
    MODEL, OPTIMIZER, N_STARTS = \
        os.path.splitext(hdf_results_file)[0].split('__')

    if MODEL == MODEL_NAME and (OPTIMIZER != 'ls_trf' or
                                MODEL == 'Fujita_SciSignal2010'):
        reader = OptimizationResultHDF5Reader(os.path.join('results',
                                                           hdf_results_file))
        result = reader.read()
        result.problem = problem

        all_results.append({
            'result': result, 'model': MODEL_NAME, 'optimizer': OPTIMIZER,
            'file': hdf_results_file
        })

cmap = cm.get_cmap('tab10')
colors = {
    legend: tuple([*cmap.colors[il], 1.0])
    for il, legend in enumerate([
        'ls_trf', 'ipopt', 'fides.subspace=full', 'fides.subspace=2D',
        'fides.subspace=full.hessian=BFGS',
        'fides.subspace=2D.hessian=BFGS',
        'fides.subspace=full.hessian=Hybrid',
        'fides.subspace=2D.hessian=Hybrid',
        'Hass2019', 'Hass2019_fmintrust'
    ])
}

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

hass2019_fmintrust_chis = np.genfromtxt(os.path.join(
    'Hass2019', f'{MODEL_NAME}_chi2s.csv',
), delimiter=',')


hass2019_fmintrust_ps = np.genfromtxt(os.path.join(
    'Hass2019', f'{MODEL_NAME}_ps.csv'
), delimiter=',')

ref = create_references(
    x=x_ref[np.asarray(
        petab_problem.x_free_indices
    )],
    fval=problem.objective(x_ref[np.asarray(petab_problem.x_free_indices)]),
    legend='Hass2019 benchmark',
    color=colors['Hass2019']
) + create_references(
    x=hass2019_fmintrust_ps[hass2019_fmintrust_chis.argmin(),
                            np.asarray(petab_problem.x_free_indices)],
    fval=problem.objective(
        hass2019_fmintrust_ps[hass2019_fmintrust_chis.argmin(),
                              np.asarray(petab_problem.x_free_indices)]
    ),
    legend='Hass2019 fmintrust',
    color=colors['Hass2019_fmintrust']
)

os.makedirs('evaluation', exist_ok=True)

all_results = sorted(
    all_results,
    key=lambda r: r['result'].optimize_result.list[0]['fval']
)

waterfall(
    [r['result'] for r in all_results],
    reference=ref,
    legends=[r['optimizer'] for r in all_results],
    colors=[colors[r['optimizer']] for r in all_results],
)
plt.tight_layout()
plt.savefig(os.path.join('evaluation',
                         f'{MODEL_NAME}_all_starts_{EVALUATION_TYPE}.pdf'))

waterfall(
    [r['result'] for r in all_results],
    reference=ref,
    legends=[r['optimizer'] for r in all_results],
    colors=[colors[r['optimizer']] for r in all_results],
    start_indices=range(int(int(N_STARTS)/10))
)
plt.tight_layout()
plt.savefig(os.path.join(
    'evaluation',
    f'{MODEL_NAME}_{int(int(N_STARTS)/10)}_starts_{EVALUATION_TYPE}.pdf'
))

dfs = [
    pd.DataFrame([
        {
            'fval': start['fval'],
            'time': start['time'],
            'ipt': start['time']/start['n_grad'],
            'iter': start['n_grad'] + start['n_sres'],
            'id': start['id'],
            'dist': np.log10(np.min(np.abs(
                start['fval'] - np.asarray([
                    s['fval']
                    for s in results['result'].optimize_result.list[:length]
                    if s['id'] != start['id']
                ])
            ))),
            'optimizer': results['optimizer']
        }
        for results in all_results
        for start in results['result'].optimize_result.list[:length]
    ]).pivot(index='id', columns=['optimizer'])
    for length in [np.min([int(N_STARTS), 300]), int(N_STARTS)]
]

df_reduced, df_full = dfs

for df in [df_full, df_reduced]:
    df.fval = df.fval - np.nanmin(df.fval) + 1
    for value in ['time', 'fval', 'iter']:
        df[value] = df[value].apply(np.log10)

df_full = df_full[np.isfinite(df_full.fval).all(axis=1)]

for df in [df_full, df_reduced]:
    df.columns = [' '.join(col).strip() for col in df.columns.values]

if EVALUATION_TYPE == 'subspace':
    for value in ['time', 'fval', 'ipt', 'dist', 'iter']:
        if value == 'dist':
            df = df_reduced
        else:
            df = df_full
        lb, ub = [
            fun([fun(df[f"{value} fides.subspace=2D"]),
                 fun(df[f"{value} fides.subspace=full"])])
            for fun in [np.nanmin, np.nanmax]
        ]
        lb -= (ub-lb)/10
        ub += (ub-lb)/10

        sns.jointplot(
            data=df,
            x=f"{value} fides.subspace=2D",
            y=f"{value} fides.subspace=full",
            kind='scatter', xlim=(lb, ub), ylim=(lb, ub),
            alpha=0.3,
            marginal_kws={'bins': 25},
        )
        plt.tight_layout()
        plt.savefig(os.path.join(
            'evaluation', f'{MODEL_NAME}_{value}_joint_{EVALUATION_TYPE}.pdf'
        )
)

        plt.subplots()

        g = sns.boxplot(data=pd.melt(df[[c for c in df.columns
                                         if c.startswith(value)]]),
                        x='variable', y='value')
        g.set_xticklabels(g.get_xticklabels(), rotation=30)
        plt.tight_layout()
        plt.savefig(os.path.join(
            'evaluation', f'{MODEL_NAME}_{value}_box_{EVALUATION_TYPE}.pdf'
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
    alpha = 1/len(result0.optimize_result.list)
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
            ttrace0.append(np.min(fvals0[:np.min([i0+1, len(fvals0)-1]) + 1]))
            ttrace1.append(np.min(fvals1[:np.min([i1+1, len(fvals1)-1]) + 1]))
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
        'evaluation', f'{MODEL_NAME}_{value}_traces_{EVALUATION_TYPE}.pdf'))

for result in all_results:
    simulation = amici.petab_objective.simulate_petab(
        petab_problem,
        model,
        problem_parameters=dict(zip(
            problem.x_names,
            result['result'].optimize_result.list[0]['x'],
        )), scaled_parameters=True,
        solver=solver
    )
    # Convert the simulation to PEtab format.
    simulation_df = amici.petab_objective.rdatas_to_simulation_df(
        simulation['rdatas'],
        model=model,
        measurement_df=petab_problem.measurement_df,
    )
    # Plot with PEtab
    petab.visualize.plot_data_and_simulation(
        exp_data=petab_problem.measurement_df,
        exp_conditions=petab_problem.condition_df,
        sim_data=simulation_df,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(
        'evaluation',
        f'{MODEL_NAME}_sim_{result["optimizer"]}_{EVALUATION_TYPE}.pdf'
    ))
