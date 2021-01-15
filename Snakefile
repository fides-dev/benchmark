import os

from benchmark import PREFIX_TEMPLATE

MODELS = ['Zheng_PNAS2012', 'Fujita_SciSignal2010', 'Boehm_JProteomeRes2014']
OPTIMIZER = ['fides.subspace=2D', 'fides.subspace=full', 'ls_trf']
N_STARTS = ['1000']


rule compile_model:
    input:
        script='compile_petab.py',
    output:
        module=os.path.join('amici_models', '{model}', '{model}', '{model}.py')
    wildcard_constraints:
        model=r'[\w_]+'
    shell:
         'python3 {input.script} {wildcards.model}'

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

rule evaluate_benchmark:
    input:
        script='evaluate.py',
        hdf5=expand(rules.run_benchmark.output.h5,
                    model=['{model}'], optimizer=OPTIMIZER, starts=N_STARTS)
    output:
        full_waterfall=expand(
            os.path.join('evaluation', '{model}_{analysis}.pdf'),
            model=['{model}'], analysis=['all_starts', 'time', 'fval', 'iter',
                                         *[f'sim_{opt}' for opt in OPTIMIZER]]
        )
    shell:
        'python3 {input.script} {wildcards.model}'

rule benchmark:
    input:
        expand(
            os.path.join('evaluation', '{model}_all_starts.pdf'),
            model=MODELS
        )
