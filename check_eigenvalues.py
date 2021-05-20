import sys

import numpy as np

from compile_petab import load_problem
from benchmark import set_solver_model_options
from evaluate import load_results
from pypesto.objective.finite_difference import FD

MODEL_NAME = sys.argv[1]
OPTIMIZER = sys.argv[2]
N_STARTS = sys.argv[3]

petab_problem, problem = load_problem(MODEL_NAME)
set_solver_model_options(problem.objective.amici_solver,
                         problem.objective.amici_model)

fd_obj = FD(
    problem.objective, hess=FD.CENTRAL, hess_via_fval=False,
    delta_grad=1e-3
)

result = load_results(MODEL_NAME, OPTIMIZER, N_STARTS)

evs = []

for start in result.optimize_result.list:
    if start.x is None:
        continue
    hess = fd_obj.get_hess(start.x)
    if np.isnan(hess).any():
        continue
    evs.append(np.min(np.linalg.eigvals(
        hess[problem.x_free_indices, :][:, problem.x_free_indices]
    )))

evs = np.real(np.asarray(evs))
np.savetxt(f'./evaluation/{MODEL_NAME}__{OPTIMIZER}__{N_STARTS}__evs.csv',
           evs, delimiter=',')

