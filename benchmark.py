import os
import sys
import petab
import fides
import pypesto.petab
import pypesto.optimize as optimize
from pypesto.objective import HistoryOptions
import pypesto.visualize as visualize
from pypesto.store import OptimizationResultHDF5Writer
import numpy as np
import matplotlib.pyplot as plt
import logging
from compile_petab import folder_base

np.random.seed(0)

PREFIX_TEMPLATE = '__'.join(['{model}', '{optimizer}', '{starts}'])

if __name__ == '__main__':
    MODEL_NAME = sys.argv[1]
    OPTIMIZER = sys.argv[2]
    N_STARTS = int(sys.argv[3])

    prefix = PREFIX_TEMPLATE.format(model=MODEL_NAME,
                                    optimizer=OPTIMIZER,
                                    starts=str(N_STARTS))

    optimizer_name = OPTIMIZER.split('.')[0]


    parsed_options = {
        option.split('=')[0]: option.split('=')[1]
        for option in OPTIMIZER.split('.')[1:]
    }

    yaml_config = os.path.join(folder_base, MODEL_NAME, MODEL_NAME + '.yaml')

    # create a petab problem
    petab_problem = petab.Problem.from_yaml(yaml_config)
    importer = pypesto.petab.PetabImporter(petab_problem)
    problem = importer.create_problem()
    problem.objective.amici_solver.setMaxSteps(int(1e3))
    problem.objective.amici_solver.setAbsoluteTolerance(1e-8)
    problem.objective.amici_solver.setRelativeTolerance(1e-8)

    if optimizer_name == 'fides':
        optim_options = {
            fides.Options.MAXITER: 1e4,
            fides.Options.THETA_MAX: 0.99,
        }

        parsed2optim = {
            'stepback': fides.Options.STEPBACK_STRAT,
            'subspace': fides.Options.SUBSPACE_DIM,
            'refine': fides.Options.REFINE_STEPBACK,
        }

        hessian_updates = {
            'BFGS': fides.BFGS(),
            'SR1': fides.SR1(),
            'FIM': None,
        }

        hessian_update = hessian_updates.get(
            parsed_options.get('hessian', 'FIM'))

        for parse_field, optim_field in parsed2optim.items():
            if parse_field in parsed_options:
                value = parsed_options[parse_field]
                if optim_field == fides.Options.REFINE_STEPBACK:
                    value = bool(value)
                optim_options[optim_field] = parsed_options[parse_field]

        optimizer = optimize.FidesOptimizer(
            options=optim_options, verbose=logging.WARNING,
            hessian_update=hessian_update
        )

    if optimizer_name == 'ls_trf':
        optimizer = optimize.ScipyOptimizer(
            method='ls_trf'
        )

    engine = pypesto.engine.SingleCoreEngine()

    options = optimize.OptimizeOptions(allow_failed_starts=True,
                                       startpoint_resample=True)

    # do the optimization

    ref = visualize.create_references(
        x=np.asarray(petab_problem.x_nominal_scaled)[np.asarray(
            petab_problem.x_free_indices
        )],
        fval=problem.objective(np.asarray(petab_problem.x_nominal_scaled)[
            np.asarray(petab_problem.x_free_indices)]
        )
    )

    print(f'Reference fval: {ref[0]["fval"]}')

    hdf_results_file = os.path.join('results', prefix + '.hdf5')
    history_options = HistoryOptions(
        trace_record=True,
        trace_record_hess=False,
        trace_record_res=False,
        trace_record_sres=False,
        trace_record_schi2=False,
    )
    result = optimize.minimize(
        problem=problem, optimizer=optimizer, n_starts=N_STARTS, engine=engine,
        options=options,
    #    history_options=history_options
    )

    visualize.waterfall(result, reference=ref, scale_y='log10')
    plt.tight_layout()
    plt.savefig(os.path.join('results', prefix + '_waterfall.pdf'))

    visualize.parameters(result, reference=ref)
    plt.tight_layout()
    plt.savefig(os.path.join('results', prefix + '_parameters.pdf'))

    visualize.optimizer_convergence(result)
    plt.tight_layout()
    plt.savefig(os.path.join('results', prefix + '_convergence.pdf'))

    #visualize.optimizer_history(result)
    #plt.tight_layout()
    #plt.savefig(os.path.join('results', prefix + '_history.pdf'))

    writer = OptimizationResultHDF5Writer(hdf_results_file)
    writer.write(result, overwrite=True)

