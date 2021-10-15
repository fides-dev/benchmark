import pypesto.petab
import petab
import os
import sys

import pandas as pd

from .evaluate import PARAMETER_ALIASES, MODEL_ALIASES

folder_base = os.path.join(os.path.dirname(__file__),
                           'Benchmark-Models-PEtab',
                           'Benchmark-Models')


def preprocess_problem(problem, model, extend_bounds):
    if model in ['Brannmark_JBC2010', 'Fiedler_BMC2016']:
        petab.flatten_timepoint_specific_output_overrides(problem)

    if extend_bounds:
        problem.parameter_df[petab.LOWER_BOUND] /= 10
        problem.parameter_df[petab.UPPER_BOUND] *= 10


def load_problem(model, force_compile=False, extend_bounds=False):
    yaml_config = os.path.join(folder_base, model, model + '.yaml')
    petab_problem = petab.Problem.from_yaml(yaml_config)
    preprocess_problem(petab_problem, model, extend_bounds)
    importer = pypesto.petab.PetabImporter(petab_problem)
    problem = importer.create_problem(force_compile=force_compile)

    matlab_model = MODEL_ALIASES.get(model, model)

    for init in ['lsqnonlin', 'fmincon']:
        try:
            plabel = pd.read_csv(os.path.join(
                'Hass2019', f'{matlab_model}_{init}_pLabel.csv'
            ))

            pstart = pd.read_csv(os.path.join(
                'Hass2019', f'{matlab_model}_{init}_ps_start.csv'
            ), names=plabel.columns[:-1])
        except:
            pass

    pnames = problem.x_names

    palias = PARAMETER_ALIASES.get(model, {})

    pnames = [
        palias.get(name, name)
        for name in pnames
    ]

    problem.x_guesses_full = pstart[pnames].values

    return petab_problem, problem


if __name__ == '__main__':
    MODEL_NAME = sys.argv[1]

    load_problem(MODEL_NAME, force_compile=True)
