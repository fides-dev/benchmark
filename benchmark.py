import os
import sys
import petab
import fides
import pypesto.petab
import pypesto.optimize as optimize
from pypesto.objective import HistoryOptions
import pypesto.visualize as visualize
from pypesto.store import (
    OptimizationResultHDF5Writer, OptimizationResultHDF5Reader
)
import numpy as np
import matplotlib.pyplot as plt
import logging
from compile_petab import folder_base

np.random.seed(0)

PREFIX_TEMPLATE = '__'.join(['{model}', '{hessian}', '{stepback}',
                             '{subspace}', '{refine}', '{starts}'])

if __name__ == '__main__':
    MODEL_NAME = sys.argv[1]
    HESSIAN_STRATEGY = sys.argv[2]
    STEPBACK_STRATEGY = sys.argv[3]
    SUBSPACE_DIM = sys.argv[4]
    REFINE = sys.argv[5]
    N_STARTS = int(sys.argv[6])

    prefix = PREFIX_TEMPLATE.format(model=MODEL_NAME,
                                    hessian=HESSIAN_STRATEGY,
                                    stepback=STEPBACK_STRATEGY,
                                    subspace=SUBSPACE_DIM,
                                    refine=str(REFINE),
                                    starts=str(N_STARTS))

    yaml_config = os.path.join(folder_base, MODEL_NAME, MODEL_NAME + '.yaml')

    # create a petab problem
    petab_problem = petab.Problem.from_yaml(yaml_config)
    importer = pypesto.petab.PetabImporter(petab_problem)
    problem = importer.create_problem()
    problem.objective.amici_solver.setMaxSteps(int(2e4))
    problem.objective.amici_solver.setAbsoluteTolerance(1e-6)
    problem.objective.amici_solver.setRelativeTolerance(1e-6)

    hessian_updates = {
        'BFGS': fides.BFGS(),
        'SR1': fides.SR1(),
        'FIM': None,
    }

    hessian_update = hessian_updates.get(HESSIAN_STRATEGY)

    optimizer = optimize.FidesOptimizer(
        options={
            fides.Options.MAXITER: 1e3,
            fides.Options.STEPBACK_STRAT: STEPBACK_STRATEGY,
            fides.Options.SUBSPACE_DIM: SUBSPACE_DIM,
            fides.Options.REFINE_STEPBACK: bool(REFINE),
            fides.Options.THETA_MAX: 0.99,
        },
        verbose=logging.WARNING, hessian_update=hessian_update
    )

    engine = pypesto.engine.SingleCoreEngine()

    options = optimize.OptimizeOptions(allow_failed_starts=True)

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
    if not os.path.exists(hdf_results_file):
        history_options = HistoryOptions(
            trace_record=True,
            trace_record_hess=False,
            trace_record_res=False,
            trace_record_sres=False,
            trace_record_schi2=False,
            storage_file=prefix + 'trace_{id}.csv',
            trace_save_iter=10
        )
        result = optimize.minimize(problem=problem, optimizer=optimizer,
                                   n_starts=N_STARTS, engine=engine,
                                   options=options)
        writer = OptimizationResultHDF5Writer(hdf_results_file)
        writer.write(result, overwrite=True)
    else:
        reader = OptimizationResultHDF5Reader(hdf_results_file)
        result = reader.read()
        result.problem = problem

    visualize.waterfall(result, reference=ref, scale_y='log10')
    plt.tight_layout()
    plt.savefig(os.path.join('results', prefix + '_waterfall.pdf'))

    visualize.parameters(result, reference=ref)
    plt.tight_layout()
    plt.savefig(os.path.join('results', prefix + '_parameters.pdf'))

    visualize.optimizer_convergence(result)
    plt.tight_layout()
    plt.savefig(os.path.join('results', prefix + '_convergence.pdf'))

