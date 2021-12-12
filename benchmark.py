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
import h5py
from compile_petab import load_problem
import scipy.optimize._lsq.trf
from typing import Dict


def set_solver_model_options(solver, model):
    solver.setMaxSteps(int(1e4))
    solver.setAbsoluteTolerance(1e-8)
    solver.setRelativeTolerance(1e-8)

    if model.getName() in ('Brannmark_JBC2010', 'Isensee_JCB2018'):
        model.setSteadyStateSensitivityMode(
            amici.SteadyStateSensitivityMode.simulationFSA
        )


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
        }

        happ = parsed_options.pop('hessian', 'FIM')
        if re.match(r'Hybrid[SB]_[0-9]+', happ):
            hybrid_happ, nswitch = happ[6:].split('_')

            happs = {'B': fides.BFGS(),
                     'S': fides.SR1()}

            hessian_update = fides.HybridFixed(
                switch_iteration=int(float(nswitch)),
                happ=happs[hybrid_happ[0]],
            )
        else:
            hessian_update = {'BFGS': fides.BFGS(),
                              'SR1': fides.SR1(),
                              'FX': fides.FX(fides.BFGS()),
                              'GNSBFGS': fides.GNSBFGS(),
                              'SSM': fides.SSM(),
                              'TSSM': fides.TSSM(),
                              'FIM': None,
                              'FIMe': None}[happ]

        for parse_field, optim_field in parsed2optim.items():
            if parse_field in parsed_options:
                value = parsed_options.pop(parse_field)
                optim_options[optim_field] = value

        if parsed_options:
            raise ValueError(f'Unknown options {parsed_options.keys()}')

        return optimize.FidesOptimizer(
            options=optim_options,
            verbose=logging.ERROR,
            hessian_update=hessian_update
        )

    if optimizer_name.startswith('ls_trf'):
        # monkeypatch xtol check
        from monkeypatch_ls_trf import trf_bounds
        scipy.optimize._lsq.trf.trf_bounds = trf_bounds

        with h5py.File(history_file, 'w') as f:
            pass

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

    engine_threads = 10
    # split parallelization for most expensive models to optimize load
    # balancing
    if MODEL_NAME in ['Bachmann_MSB2011', 'Isensee_JCB2018',
                      'Lucarelli_CellSystems2018', 'Beer_MolBioSystems2014']:
        # Bachmann   nc =  36 (4* 9)
        # Lucarelli  nc =  16 (4* 4)
        # Isensee    nc = 123 (4*30 + 3)
        # Beer       nc =  19 (4* 4 + 3)
        engine_threads = 3
        objective.n_threads = 4

    engine = pypesto.engine.MultiThreadEngine(n_threads=engine_threads)

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

