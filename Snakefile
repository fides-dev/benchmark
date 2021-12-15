import os

from benchmark import PREFIX_TEMPLATE
from evaluate import OPTIMIZER_FORWARD, N_STARTS_FORWARD
from comparison import MODELS


rule compile_model:
    input:
        script='compile_petab.py',
    output:
        module=os.path.join('amici_models', '{model}', '{model}', '{model}.py')
    wildcard_constraints:
        model=r'[\w_]+'
    shell:
         'python3 {input.script} {wildcards.model}'

rule run_benchmark_very_long:
    input:
        model=os.path.join('amici_models', '{model}', '{model}', '{model}.py')
    output:
        h5=os.path.join('results', PREFIX_TEMPLATE.format(
            model='{model}', optimizer='{optimizer}', starts='{starts}'
        ) + '.hdf5'),
        stats=os.path.join('stats', PREFIX_TEMPLATE.format(
            model='{model}', optimizer='{optimizer}', starts='{starts}'
        ) + '__STATS.hdf5')
    wildcard_constraints:
        model='(Beer_MolBioSystems2014|Isensee_JCB2018|Lucarelli_CellSystems2018)',
    shell:
         'python3 benchmark.py {wildcards.model} {wildcards.optimizer} '
         '{wildcards.starts}'


rule run_benchmark_extended:
    input:
        model=os.path.join('amici_models', '{model}', '{model}', '{model}.py')
    output:
        h5=os.path.join('results', PREFIX_TEMPLATE.format(
            model='{model}', optimizer='{optimizer}', starts='{starts}'
        ) + '.hdf5'),
        stats=os.path.join('stats', PREFIX_TEMPLATE.format(
            model='{model}', optimizer='{optimizer}', starts='{starts}'
        ) + '__STATS.hdf5')
    wildcard_constraints:
        model='(Beer_MolBioSystems2014|Isensee_JCB2018|Bachmann_MSB2011|'
              'Lucarelli_CellSystems2018|Weber_BMC2015)',
        optimizer='(fides\.subspace=2D\.hessian=FX|'
                  'fides\.subspace=2D\.hessian=GNSBFGS|'
                  'fides\.subspace=2D\.hessian=SSM|'
                  'fides\.subspace=2D\.hessian=TSSM|'
                  'fides\.subspace=2D\.ebounds=1000|'
                  'fides\.subspace=2D\.ebounds=100|'
                  'fides\.subspace=2D\.ebounds=Inf)'
    shell:
         'python3 benchmark.py {wildcards.model} {wildcards.optimizer} '
         '{wildcards.starts}'


rule run_benchmark_long:
    input:
        model=os.path.join('amici_models', '{model}', '{model}', '{model}.py')
    output:
        h5=os.path.join('results', PREFIX_TEMPLATE.format(
            model='{model}', optimizer='{optimizer}', starts='{starts}'
        ) + '.hdf5'),
        stats=os.path.join('stats', PREFIX_TEMPLATE.format(
            model='{model}', optimizer='{optimizer}', starts='{starts}'
        ) + '__STATS.hdf5')
    wildcard_constraints:
        model='(Bachmann_MSB2011|Weber_BMC2015)'
    shell:
         'python3 benchmark.py {wildcards.model} {wildcards.optimizer} '
         '{wildcards.starts}'

rule run_benchmark_short:
    input:
        model=os.path.join('amici_models', '{model}', '{model}', '{model}.py')
    output:
        h5=os.path.join('results', PREFIX_TEMPLATE.format(
            model='{model}', optimizer='{optimizer}', starts='{starts}'
        ) + '.hdf5'),
        stats=os.path.join('stats', PREFIX_TEMPLATE.format(
            model='{model}', optimizer='{optimizer}', starts='{starts}'
        ) + '__STATS.hdf5')
    wildcard_constraints:
        model='(Crauste_CellSystems2017|Boehm_JProteomeRes2014|Bruno_JExpBot2016)'
    shell:
         'python3 benchmark.py {wildcards.model} {wildcards.optimizer} '
         '{wildcards.starts}'

rule run_benchmark:
    input:
        model=os.path.join('amici_models', '{model}', '{model}', '{model}.py')
    output:
        h5=os.path.join('results', PREFIX_TEMPLATE.format(
            model='{model}', optimizer='{optimizer}', starts='{starts}'
        ) + '.hdf5'),
        stats=os.path.join('stats', PREFIX_TEMPLATE.format(
            model='{model}', optimizer='{optimizer}', starts='{starts}'
        ) + '__STATS.hdf5')
    shell:
         'python3 benchmark.py {wildcards.model} {wildcards.optimizer} '
         '{wildcards.starts}'

rule check_eigenvalues:
    input:
        script='check_eigenvalues.py',
        model=os.path.join('amici_models', '{model}', '{model}', '{model}.py'),
    output:
        csv=os.path.join('evaluation', '{model}__{optimizer}__{starts}__evs.csv'),
    shell:
         'python3 {input.script} {wildcards.model} {wildcards.optimizer} '
         '{wildcards.starts}'

rule evaluate_subspace_benchmark:
    input:
        script='evaluate.py',
        hdf5=expand(rules.run_benchmark.output.h5,
                    model=['{model}'], optimizer=OPTIMIZER_FORWARD,
                    starts=N_STARTS_FORWARD)
    output:
        full_waterfall=os.path.join('evaluation', '{model}_all_starts.pdf')
    shell:
        'python3 {input.script} {wildcards.model}'

rule benchmark:
    input:
        expand(
            os.path.join('evaluation', '{model}_all_starts.pdf'),
            model=MODELS
        )

rule eigenvalues:
    input:
        expand(rules.check_eigenvalues.output.csv,
               model=MODELS,
               optimizer=[
                    'fides.subspace=2D',
                    'fides.subspace=full',
                    'fides.subspace=full.hessian=BFGS',
                    'fides.subspace=2D.hessian=BFGS',
                    'fides.subspace=full.hessian=SR1',
                    'fides.subspace=2D.hessian=SR1',
                    'ls_trf',
                    'ls_trf_2D',
                    'fmincon',
                    'lsqnonlin',
               ],
               starts=N_STARTS_FORWARD)

ruleorder: run_benchmark_extended > run_benchmark_very_long > run_benchmark_long > run_benchmark_short > run_benchmark
