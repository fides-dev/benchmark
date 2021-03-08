import os

from benchmark import PREFIX_TEMPLATE

MODELS_SUBSPACE = ['Zheng_PNAS2012', 'Fujita_SciSignal2010',
                   'Boehm_JProteomeRes2014']
OPTIMIZER_SUBSPACE = ['fides.subspace=2D', 'fides.subspace=full',
                      'fides.subspace=2D.hessian=Hybrid',
                      'fides.subspace=full.hessian=Hybrid'
                      'fides.subspace=full.hessian=BFGS',
                      'fides.subspace=2D.hessian=BFGS', 'ls_trf']
N_STARTS_SUBSPACE = ['1000']

MODELS_ADJOINT = ['Chen_MSB2009']
OPTIMIZER_ADJOINT = ['fides.subspace=full.hessian=BFGS',
                     'fides.subspace=2D.hessian=BFGS',
                     'ipopt']
N_STARTS_ADJOINT = ['100']


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

rule evaluate_subspace_benchmark:
    input:
        script='evaluate.py',
        hdf5=expand(rules.run_benchmark.output.h5,
                    model=['{model}'], optimizer=OPTIMIZER_SUBSPACE,
                    starts=N_STARTS_SUBSPACE)
    output:
        full_waterfall=expand(
            os.path.join('evaluation', '{model}_{analysis}_subspace.pdf'),
            model=['{model}'], analysis=['all_starts', 'time', 'fval', 'iter',
                                         *[f'sim_{opt}' for opt
                                           in OPTIMIZER_SUBSPACE]]
        )
    shell:
        'python3 {input.script} {wildcards.model} subspace'

rule evaluate_adjoint_benchmark:
    input:
        script='evaluate.py',
        hdf5=expand(rules.run_benchmark.output.h5,
                    model=['{model}'], optimizer=OPTIMIZER_ADJOINT,
                    starts=N_STARTS_ADJOINT)
    output:
        full_waterfall=expand(
            os.path.join('evaluation', '{model}_{analysis}_adjoint.pdf'),
            model=['{model}'], analysis=['all_starts', 'time', 'fval', 'iter',
                                         *[f'sim_{opt}' for opt
                                           in OPTIMIZER_ADJOINT]]
        )
    shell:
        'python3 {input.script} {wildcards.model} adjoint'

rule benchmark:
    input:
        expand(
            os.path.join('evaluation', '{model}_all_starts_subspace.pdf'),
            model=MODELS_SUBSPACE
        ),
        expand(
            os.path.join('evaluation', '{model}_all_starts_adjoint.pdf'),
            model=MODELS_ADJOINT
        ),

