import os
import sys
import amici
import fides
import pypesto
import re
import distutils.util
import pypesto.optimize as optimize
import pypesto.visualize as visualize
from pypesto.store import OptimizationResultHDF5Writer
import numpy as np
import matplotlib.pyplot as plt
import logging
from compile_petab import load_problem
import scipy.optimize._lsq.trf
from typing import Dict


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

    if model.getName() in ('Brannmark_JBC2010', 'Isensee_JCB2018'):
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


def get_optimizer(optimizer_name: str, history_file: str,
                  parsed_options: Dict):
    if optimizer_name == 'fides':
        optim_options = {
            fides.Options.MAXITER: MAX_ITER,
            fides.Options.FATOL: 0.0,
            fides.Options.FRTOL: 0.0,
            fides.Options.XTOL: 1e-6,
            fides.Options.GATOL: 0.0,
            fides.Options.GRTOL: 0.0,
            fides.Options.HISTORY_FILE: history_file
        }

        parsed2optim = {
            'stepback': fides.Options.STEPBACK_STRAT,
            'subspace': fides.Options.SUBSPACE_DIM,
            'refine': fides.Options.REFINE_STEPBACK,
            'scaled_gradient': fides.Options.SCALED_GRADIENT,
            'restrict': fides.Options.RESTRICT_HESS_APPROX,
        }

        happ = parsed_options.pop('hessian', 'FIM')
        enforce_curv = bool(distutils.util.strtobool(
            parsed_options.pop('enforce_curv', 'True')
        ))

        if re.match(r'Hybrid[SB]0?_[0-9]+', happ):
            hybrid_happ, nswitch = happ[6:].split('_')

            happs = {
                'B': fides.BFGS(init_with_hess=hybrid_happ.endswith('0'),
                                enforce_curv_cond=enforce_curv),
                'S': fides.SR1(init_with_hess=hybrid_happ.endswith('0'))
            }

            hessian_update = fides.HybridFixed(
                switch_iteration=int(float(nswitch)),
                happ=happs[hybrid_happ[0]],
            )
        elif re.match(r'HybridF[SB]0?_[0-9]+', happ):
            hybrid_happ, tswitch = happ[7:].split('_')

            happs = {
                'B': fides.BFGS(init_with_hess=hybrid_happ.endswith('0'),
                                enforce_curv_cond=enforce_curv),
                'S': fides.SR1(init_with_hess=hybrid_happ.endswith('0'))
            }

            hessian_update = fides.HybridFrac(
                switch_threshold=float(tswitch.replace('-', '.')),
                happ=happs[hybrid_happ[0]],
            )
        else:
            hessian_update = {
                'BFGS': fides.BFGS(enforce_curv_cond=enforce_curv),
                'SR1': fides.SR1(),
                'FX': fides.FX(fides.BFGS(enforce_curv_cond=enforce_curv)),
                'GNSBFGS': fides.GNSBFGS(enforce_curv_cond=enforce_curv),
                'SSM': fides.SSM(enforce_curv_cond=enforce_curv),
                'TSSM': fides.TSSM(enforce_curv_cond=enforce_curv),
                'FIM': None,
                'FIMe': None,
            }[happ]

        for parse_field, optim_field in parsed2optim.items():
            if parse_field in parsed_options:
                value = parsed_options.pop(parse_field)
                if optim_field in [fides.Options.REFINE_STEPBACK,
                                   fides.Options.SCALED_GRADIENT,
                                   fides.Options.RESTRICT_HESS_APPROX]:
                    value = bool(distutils.util.strtobool(value))
                optim_options[optim_field] = value

        if parsed_options:
            raise ValueError(f'Unknown options {parsed_options.keys()}')

        return optimize.FidesOptimizer(
            options=optim_options, verbose=logging.ERROR,
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

    raise ValueError('Unknown optimizer name.')


np.random.seed(0)

PREFIX_TEMPLATE = '__'.join(['{model}', '{optimizer}', '{starts}'])
MAX_ITER = 1e5

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

    petab_problem, problem = load_problem(
        MODEL_NAME, extend_bounds=float(parsed_options.pop('ebounds', 1.0))
    )

    if isinstance(problem.objective, pypesto.AmiciObjective):
        objective = problem.objective
    else:
        objective = problem.objective._objectives[0]

    if optimizer_name.startswith('ls_trf') or \
            parsed_options.get('hessian', 'FIM') in ('FIMe', 'FX', 'GNSBFGS',
                                                     'SSM', 'TSSM'):
        objective.amici_model.setAddSigmaResiduals(True)
    set_solver_model_options(objective.amici_solver,
                             objective.amici_model)

    objective.guess_steadystate = False

    os.makedirs('stats', exist_ok=True)

    optimizer = get_optimizer(
        optimizer_name,
        os.path.join('stats', PREFIX_TEMPLATE.format(
            model=MODEL_NAME, optimizer=OPTIMIZER, starts=str(N_STARTS)
        ) + '__STATS.hdf5'),
        parsed_options
    )

    engine = pypesto.engine.MultiThreadEngine(n_threads=10)

    options = optimize.OptimizeOptions(allow_failed_starts=True,
                                       startpoint_resample=False,
                                       report_sres=False, report_hess=False)

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

    result = optimize.minimize(
        problem=problem, optimizer=optimizer, n_starts=N_STARTS,
        engine=engine,
        options=options, progress_bar=False, filename=None,
    )

    visualize.waterfall(result, reference=ref, scale_y='log10')
    plt.tight_layout()
    plt.savefig(os.path.join('results', prefix + '_waterfall.pdf'))

    writer = OptimizationResultHDF5Writer(hdf_results_file)
    writer.write(result, overwrite=True)

