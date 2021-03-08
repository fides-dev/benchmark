import os
import sys
import petab
import amici
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
MAX_ITER = 1e3
MAX_TIME = 7200.0

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
    if MODEL_NAME == 'Chen_MSB2009':
        # don't estimate parameters on linear scale
        petab_problem.parameter_df.loc[
            petab_problem.parameter_df[petab.PARAMETER_SCALE] == petab.LIN,
            petab.ESTIMATE
        ] = 0

    importer = pypesto.petab.PetabImporter(petab_problem)
    problem = importer.create_problem()

    problem.objective.amici_solver.setMaxSteps(int(1e4))
    problem.objective.amici_solver.setAbsoluteTolerance(1e-8)
    problem.objective.amici_solver.setRelativeTolerance(1e-8)

    if MODEL_NAME == 'Chen_MSB2009':
        problem.objective.amici_solver.setMaxSteps(int(2e5))
        problem.objective.amici_solver.setInterpolationType(
            amici.InterpolationType_polynomial
        )

    if MODEL_NAME == 'Fujita_SciSignal2010':
        problem.objective.amici_solver.setMaxSteps(int(2e4))

    if optimizer_name == 'fides':
        optim_options = {
            fides.Options.MAXITER: MAX_ITER,
            fides.Options.MAXTIME: MAX_TIME,
        }

        parsed2optim = {
            'stepback': fides.Options.STEPBACK_STRAT,
            'subspace': fides.Options.SUBSPACE_DIM,
            'refine': fides.Options.REFINE_STEPBACK,
        }

        hessian_updates = {
            'Hybrid_05': fides.HybridUpdate(
                switch_iteration=np.ceil(0.5*problem.dim)),
            'Hybrid_1': fides.HybridUpdate(switch_iteration=problem.dim),
            'Hybrid_2': fides.HybridUpdate(switch_iteration=2*problem.dim),
            'Hybrid_5': fides.HybridUpdate(switch_iteration=5*problem.dim),
            'BFGS': fides.BFGS(),
            'SR1': fides.SR1(),
            'FIM': None,
        }

        if parsed_options.get('hessian', 'FIM') not in ['FIM', 'Hybrid']:
            hessian_update = hessian_updates.get(parsed_options.get('hessian'))
            if MODEL_NAME == 'Chen_MSB2009':
                problem.objective.amici_solver.setSensitivityMethod(
                    amici.SensitivityMethod.adjoint
                )
        else:
            hessian_update = hessian_updates.get(parsed_options.get('hessian',
                                                                    'FIM'))

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
            method='ls_trf', options={'max_nfev': MAX_ITER,
                                      'xtol': 0.0,
                                      'ftol': 1e-8,
                                      'gtol': 1e-6}
        )

    if optimizer_name == 'ipopt':
        problem.objective.amici_solver.setSensitivityMethod(
            amici.SensitivityMethod.adjoint
        )
        optimizer = optimize.IpoptOptimizer(
            options={'max_iter': int(MAX_ITER),
                     'tol': 1e-8,
                     'acceptable_tol': 1e-100,
                     'max_cpu_time': MAX_TIME}
        )

    engine = pypesto.engine.MultiThreadEngine(n_threads=10)

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

    if MODEL_NAME == 'Chen_MSB2009':
        history_options = HistoryOptions(
            trace_record=True,
            trace_record_hess=False,
            trace_record_res=False,
            trace_record_sres=False,
            trace_record_schi2=False,
            trace_save_iter=10,
            storage_file=os.path.join(
                'results',
                f'{MODEL_NAME}__{OPTIMIZER}__{N_STARTS}__trace{{id}}.csv'
            )
        )
    else:
        history_options=None
    result = optimize.minimize(
        problem=problem, optimizer=optimizer, n_starts=N_STARTS,
        engine=engine,
        options=options,
        history_options=history_options
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

