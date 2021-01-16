import sys
import os

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

from compile_petab import folder_base


MODEL_NAME = sys.argv[1]

hdf5_files = [r for r in os.listdir('results')
              if r.startswith(MODEL_NAME) and r.endswith('.hdf5')]

yaml_config = os.path.join(folder_base, MODEL_NAME, MODEL_NAME + '.yaml')
petab_problem = petab.Problem.from_yaml(yaml_config)
importer = pypesto.petab.PetabImporter(petab_problem)
problem = importer.create_problem()
model = importer.create_model()

all_results = []
for hdf_results_file in hdf5_files:
    MODEL, OPTIMIZER, N_STARTS = \
        hdf_results_file.split('__')

    if MODEL == MODEL_NAME and OPTIMIZER != 'ls_trf':
        reader = OptimizationResultHDF5Reader(os.path.join('results',
                                                           hdf_results_file))
        result = reader.read()
        result.problem = problem

        all_results.append({
            'result': result, 'model': MODEL_NAME, 'optimizer': OPTIMIZER,
            'file': hdf_results_file
        })

ref = create_references(
    x=np.asarray(petab_problem.x_nominal_scaled)[np.asarray(
        petab_problem.x_free_indices
    )],
    fval=problem.objective(np.asarray(petab_problem.x_nominal_scaled)[
        np.asarray(petab_problem.x_free_indices)]
    )
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
)
plt.tight_layout()
plt.savefig(os.path.join('evaluation', f'{MODEL_NAME}_all_starts.pdf'))

df = pd.DataFrame([
    {
        'fval': start['fval'],
        'time': start['time'],
        'iter': start['n_fval'],
        'itertime': start['n_fval']/start['time'],
        'id': start['id'],
        'optimizer': results['optimizer']
    }
    for results in all_results
    for start in results['result'].optimize_result.list
])

df = df.pivot(index='id', columns=['optimizer'])

df.fval = df.fval - np.nanmin(df.fval) + 1
for value in ['time', 'fval', 'iter', 'itertime']:
    df[value] = df[value].apply(np.log10)
df = df[np.isfinite(df.fval).all(axis=1)]

df.columns = [' '.join(col).strip() for col in df.columns.values]

for value in ['time', 'fval', 'iter', 'itertime']:
    lb, ub = [
        fun([fun(df[f"{value} fides.subspace=2D"]),
             fun(df[f"{value} fides.subspace=full"])])
        for fun in [np.nanmin, np.nanmax]
    ]
    lb -= (ub-lb)/10
    ub += (ub-lb)/10

    g = sns.jointplot(data=df,
                      x=f"{value} fides.subspace=2D",
                      y=f"{value} fides.subspace=full",
                      kind='reg',
                      xlim=(lb, ub), ylim=(lb, ub),
                      marginal_kws={'bins': 25},
                      joint_kws={'scatter_kws': {'alpha': 0.3}})
    plt.tight_layout()
    plt.savefig(os.path.join('evaluation', f'{MODEL_NAME}_{value}.pdf'))
    plt.show()

for result in all_results:
    simulation = amici.petab_objective.simulate_petab(
        petab_problem,
        model,
        problem_parameters=dict(zip(
            problem.x_names,
            result['result'].optimize_result.list[0]['x'],
        )), scaled_parameters=True
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
    plt.show()
    plt.tight_layout()
    plt.savefig(os.path.join('evaluation',
                             f'{MODEL_NAME}_sim_{result["optimizer"]}.pdf'))

