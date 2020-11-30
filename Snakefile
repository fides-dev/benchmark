import os

from benchmark import PREFIX_TEMPLATE

MODELS = ['Zheng_PNAS2012', 'Fujita_SciSignal2010', 'Boehm_JProteomeRes2014']
HESSIAN = ['BFGS', 'SR1', 'FIM']
STEPBACK = ['reflect_single', 'reflect', 'truncate', 'mixed']
SUBSPACE = ['2D', 'full']
REFINE = ['0', '1']
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
            model='{model}', hessian='{hessian}', stepback='{stepback}',
            subspace='{subspace}', refine='{refine}', starts='{starts}'
        ) + '.hdf5')
    shell:
         'python3 {input.script} {wildcards.model} {wildcards.hessian} '
         '{wildcards.stepback} {wildcards.subspace} {wildcards.refine} '
         '{wildcards.starts}'

rule evaluate_benchmark:
    input:
        script='evaluate.py',
        hdf5=expand(rules.run_benchmark.output.h5,
                    model=['{model}'], hessian=HESSIAN, stepback=STEPBACK,
                    subspace=SUBSPACE, refine=REFINE, starts=N_STARTS)
    output:
        full_waterfall=expand(
            os.path.join('evaluation', '{model}_{analysis}.pdf'),
            model=['{model}'], analysis=['all_starts', 'by_hess', 'by_refine', 'by_stepback', 'by_subspace']
        )
    shell:
        'python3 {input.script} {wildcards.model}'

rule benchmark:
    input:
        expand(
            os.path.join('evaluation', '{model}_all_starts.pdf'),
            model=MODELS
        )
