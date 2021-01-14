import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm

import pypesto.petab
import petab
from pypesto.store import OptimizationResultHDF5Reader
from pypesto.visualize import waterfall, create_references, \
    optimization_run_properties_per_multistart

from compile_petab import folder_base


MODEL_NAME = sys.argv[1]

hdf5_files = [r for r in os.listdir('results')
              if r.startswith(MODEL_NAME) and r.endswith('.hdf5')]

yaml_config = os.path.join(folder_base, MODEL_NAME, MODEL_NAME + '.yaml')
petab_problem = petab.Problem.from_yaml(yaml_config)
importer = pypesto.petab.PetabImporter(petab_problem)
problem = importer.create_problem()

all_results = []
for hdf_results_file in hdf5_files:
    MODEL_NAME, HESSIAN, STEPBACK, SUBSPACE, REFINE, N_STARTS = \
        hdf_results_file.split('__')

    if HESSIAN == 'FIM' and REFINE == '0' and STEPBACK == 'reflect_single':
        reader = OptimizationResultHDF5Reader(os.path.join('results',
                                                           hdf_results_file))
        result = reader.read()
        result.problem = problem

        all_results.append({
            'result': result, 'model': MODEL_NAME, 'hess': HESSIAN,
            'stepback': STEPBACK, 'subspace': SUBSPACE, 'refine': REFINE,
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

tab10 = cm.get_cmap('tab20')
colors = {}
for field in ['hess', 'stepback', 'refine']:
    opts = np.unique([r[field] for r in all_results])
    colors[field] = {**{
        ('full', opt): (*tab10.colors[iopt * 2], 1.0)
        for iopt, opt in enumerate(opts)
    }, **{
        ('2D', opt): (*tab10.colors[iopt * 2 + 1], 1.0)
        for iopt, opt in enumerate(opts)
    }}

waterfall(
    [r['result'] for r in sorted(
        all_results,
        key=lambda r: r['result'].optimize_result.list[0]['fval']
    )],
    legends=[os.path.splitext(r['file'])[0] for r in all_results],
)
plt.tight_layout()
plt.savefig(os.path.join('evaluation', f'{MODEL_NAME}_all_starts.pdf'))

df = pd.DataFrame([
    {'fval': start['fval'], 'time': start['time'], 'iter': start['n_fval'],
     'hess': results['hess'], 'stepback': results['stepback'],
     'subspace': results['subspace'], 'refine': results['refine']}
    for results in all_results
    for start in results['result'].optimize_result.list
])

df.fval = df.fval - np.nanmin(df.fval) + 1
df.fval = df.fval.apply(np.log10)
df.time = df.time.apply(np.log10)
df.iter = df.iter.apply(np.log10)
df = df[np.isfinite(df.fval)]

df.rename(columns={'fval': 'log10(fval - minfval + 1)',
                   'time': 'log10(time)',
                   'iter': 'log10(iter)'}, inplace=True)

df = pd.melt(df, value_vars=['log10(time)', 'log10(fval - minfval + 1)',
                             'log10(iter)'],
             id_vars=['hess', 'stepback', 'subspace', 'refine'])


for refine in df.refine.unique():
    for variable in df.variable.unique():
        g = sns.FacetGrid(
            df[(df.refine == refine) & (df.variable == variable)],
            row='hess', col='stepback', sharey=False
        )
        g.map(sns.boxplot, 'subspace', "value")
        g.set_xticklabels(rotation=90)
        g.set_ylabels(variable)
        plt.tight_layout()
        plt.savefig(os.path.join(
            'evaluation',
            f'{MODEL_NAME}_refine{refine}_variable{variable}.pdf'
        ))
        plt.show()


"""
for field in ['hess', 'stepback', 'refine']:
    waterfall(
        [r['result'] for r in all_results],
        colors=np.asarray([colors[field][(r['subspace'], r[field])]
                           for r in all_results]),
        legends=[f'{r["subspace"]}/{r[field]}' for r in all_results],
        reference=ref,
    )
    optimization_run_properties_per_multistart(
        [r['result'] for r in all_results],
        properties_to_plot=['time', 'n_fval'],
        colors=[colors[field][(r['subspace'], r[field])] for r in all_results],
        legends=[f'{r["subspace"]}/{r[field]}' for r in all_results],
    )
    plt.tight_layout()
    plt.savefig(os.path.join('evaluation', f'{MODEL_NAME}_by_{field}.pdf'))
"""

