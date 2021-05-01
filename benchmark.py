import os
import sys
import amici
import fides
import re
import distutils.util
import pypesto
import pypesto.optimize as optimize
from pypesto.objective import HistoryOptions
import pypesto.visualize as visualize
from pypesto.store import OptimizationResultHDF5Writer
import numpy as np
import matplotlib.pyplot as plt
import logging
from compile_petab import load_problem
import scipy.optimize._lsq.trf


def set_solver_model_options(solver, model):
    solver.setMaxSteps(int(1e4))
    solver.setAbsoluteTolerance(1e-8)
    solver.setRelativeTolerance(1e-8)

    if model.getName() == 'Chen_MSB2009':
        solver.setMaxSteps(int(2e5))
        solver.setInterpolationType(
            amici.InterpolationType_polynomial
        )
        solver.setSensitivityMethod(
            amici.SensitivityMethod.adjoint
        )

    if model.getName() == 'Brannmark_JBC2010':
        model.setSteadyStateSensitivityMode(
            amici.SteadyStateSensitivityMode.simulationFSA
        )


def check_termination(dF, F, dx_norm, x_norm, ratio, ftol, xtol):
    """
    Check termination condition for nonlinear least squares.
    Custom monkeypatch implementation that fixes xtol check
    """
    ftol_satisfied = dF < ftol * F and ratio > 0.25
    xtol_satisfied = dx_norm < xtol * (1 + x_norm)

    if ftol_satisfied and xtol_satisfied:
        return 4
    elif ftol_satisfied:
        return 2
    elif xtol_satisfied:
        return 3
    else:
        return None


def get_optimizer(optimizer_name: str):
    if optimizer_name == 'fides':
        optim_options = {
            fides.Options.MAXITER: MAX_ITER,
            fides.Options.MAXTIME: MAX_TIME,
            fides.Options.FATOL: 0.0,
            fides.Options.FRTOL: 0.0,
            fides.Options.XTOL: 1e-6,
            fides.Options.GATOL: 0.0,
            fides.Options.GRTOL: 0.0,
        }

        parsed2optim = {
            'stepback': fides.Options.STEPBACK_STRAT,
            'subspace': fides.Options.SUBSPACE_DIM,
            'refine': fides.Options.REFINE_STEPBACK,
            'scaled_gradient': fides.Options.SCALED_GRADIENT,
        }

        happ = parsed_options.get('hessian', 'FIM')

        if re.match(r'Hybrid(S|B)0?_[0-9]+', happ):
            hybrid_happ, ndim = happ[6:].split('_')
            fides.HybridUpdate(switch_iteration=float(ndim) * problem.dim,
                               happ={'B': fides.BFGS(),
                                     'S': fides.SR1()}.get(hybrid_happ[0]),
                               init_with_hess=hybrid_happ.endswith('0'))
        else:
            hessian_update = {
                'BFGS': fides.BFGS(),
                'SR1': fides.SR1(),
                'FIM': None,
                'FIMe': None,
            }.get(happ)

        for parse_field, optim_field in parsed2optim.items():
            if parse_field in parsed_options:
                value = parsed_options[parse_field]
                if optim_field in [fides.Options.REFINE_STEPBACK,
                                   fides.Options.SCALED_GRADIENT]:
                    value = bool(distutils.util.strtobool(value))
                optim_options[optim_field] = value

        return optimize.FidesOptimizer(
            options=optim_options, verbose=logging.WARNING,
            hessian_update=hessian_update
        )

    if optimizer_name.startswith('ls_trf'):
        # monkeypatch xtol check
        scipy.optimize._lsq.trf.check_termination = check_termination

        options = {'max_nfev': MAX_ITER,
                   'xtol': 1e-6,
                   'ftol': 0.0,
                   'gtol': 0.0}
        if optimizer_name == 'ls_trf_2D':
            options['tr_solver'] = 'lsmr'
        elif optimizer_name == 'ls_trf':
            options['tr_solver'] = 'exact'

        return optimize.ScipyOptimizer(
            method='ls_trf', options=options
        )

    if optimizer_name == 'ipopt':
        return optimize.IpoptOptimizer(
            options={'max_iter': int(MAX_ITER),
                     'tol': 1e-8,
                     'acceptable_tol': 1e-100,
                     'max_cpu_time': MAX_TIME}
        )


np.random.seed(0)

PREFIX_TEMPLATE = '__'.join(['{model}', '{optimizer}', '{starts}'])
MAX_ITER = 1e4
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

    petab_problem, problem = load_problem(MODEL_NAME)
    if optimizer_name.startswith('ls_trf') or \
            parsed_options.get('hessian', 'FIM') == 'FIMe':
        problem.objective.amici_model.setAddSigmaResiduals(True)
    set_solver_model_options(problem.objective.amici_solver,
                             problem.objective.amici_model)

    problem.objective.guess_steadystate = False

    optimizer = get_optimizer(optimizer_name)

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
        history_options = None
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

    writer = OptimizationResultHDF5Writer(hdf_results_file)
    writer.write(result, overwrite=True)

