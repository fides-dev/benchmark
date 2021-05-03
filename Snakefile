import os

from benchmark import PREFIX_TEMPLATE
from evaluate import OPTIMIZER_FORWARD, OPTIMIZER_ADJOINT , N_STARTS_ADJOINT, \
    N_STARTS_FORWARD

MODELS_FORWARD = ['Boehm_JProteomeRes2014']

MODELS_ADJOINT = []

rule compile_model:
    input:
        script='compile_petab.py',
    output:
        module=os.path.join('amici_models', '{model}', '{model}', '{model}.py')
    wildcard_constraints:
        model=r'[\w_]+'
    shell:
         'python3 {input.script} {wildcards.model}'

rule run_benchmark_long:
    input:
        script='benchmark.py',
        model=os.path.join('amici_models', '{model}', '{model}', '{model}.py')
    output:
        h5=os.path.join('results', PREFIX_TEMPLATE.format(
            model='{model}', optimizer='{optimizer}', starts='{starts}'
        ) + '.hdf5')
    wildcard_constraints:
        model='(Chen_MSB2009|Beer_MolBioSystems2014)'
    shell:
         'python3 {input.script} {wildcards.model} {wildcards.optimizer} '
         '{wildcards.starts}'

rule run_benchmark_short:
    input:
        script='benchmark.py',
        model=os.path.join('amici_models', '{model}', '{model}', '{model}.py')
    output:
        h5=os.path.join('results', PREFIX_TEMPLATE.format(
            model='{model}', optimizer='{optimizer}', starts='{starts}'
        ) + '.hdf5')
    wildcard_constraints:
        model='(Crauste_CellSystems2017|Boehm_JProteomeRes2014)'
    shell:
         'python3 {input.script} {wildcards.model} {wildcards.optimizer} '
         '{wildcards.starts}'

rule run_benchmark:
    input:
        script='benchmark.py',
        model=os.path.join('amici_models', '{model}', '{model}', '{model}.py')
    output:
        h5=os.path.join('results', PREFIX_TEMPLATE.format(
            model='{model}', optimizer='{optimizer}', starts='{starts}'
        ) + '.hdf5')
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
        full_waterfall=os.path.join('evaluation',
                                    '{model}_all_starts_forward.pdf')
    shell:
        'python3 {input.script} {wildcards.model} subspace'

rule evaluate_adjoint_benchmark:
    input:
        script='evaluate.py',
        hdf5=expand(rules.run_benchmark.output.h5,
                    model=['{model}'], optimizer=OPTIMIZER_ADJOINT,
                    starts=N_STARTS_ADJOINT)
    output:
        full_waterfall=os.path.join('evaluation',
                                    '{model}_all_starts_adjoint.pdf')
    shell:
        'python3 {input.script} {wildcards.model} adjoint'

rule benchmark:
    input:
        expand(
            os.path.join('evaluation', '{model}_all_starts_forward.pdf'),
            model=MODELS_FORWARD
        ),
        expand(
            os.path.join('evaluation', '{model}_all_starts_adjoint.pdf'),
            model=MODELS_ADJOINT
        ),

ruleorder: run_benchmark_long > run_benchmark_short > run_benchmark
