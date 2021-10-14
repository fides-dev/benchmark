import pypesto.petab
import petab
import os
import sys

import pandas as pd

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

    matlab_model = {
        'Crauste_CellSystems2017': 'Crauste_ImmuneCells_CellSystems2017',
        'Bruno_JExpBot2016': 'Bruno_Carotines_JExpBio2016',
        'Schwen_PONE2014': 'Schwen_InsulinMouseHepatocytes_PlosOne2014',

    }.get(model, model)

    plabel = pd.read_csv(os.path.join('Hass2019',
                                      f'{matlab_model}_fmincon_pLabel.csv'))

    pstart = pd.read_csv(os.path.join('Hass2019',
                                      f'{matlab_model}_fmincon_ps_start.csv'),
                         names=plabel.columns[:-1])

    pnames = problem.x_names
    if model == 'Fujita_SciSignal2010':
        palias = {'init_AKT': 'init_Akt',
                  'scaling_pAkt_tot': 'scaleFactor_pAkt',
                  'scaling_pEGFR_tot': 'scaleFactor_pEGFR',
                  'scaling_pS6_tot': 'scaleFactor_pS6'}

    elif model == 'Zheng_PNAS2012':
        palias = {'sigma': 'noise'}

    elif model == 'Bruno_JExpBot2016':
        palias = {'init_b10_1': 'init_b10',
                  'init_bcry_1': 'init_bcar1',
                  'init_zea_1': 'init_bcry',
                  'init_ohb10_1': 'init_ohb10'}

    else:
        palias = {}

    pnames = [
        palias.get(name, name)
        for name in pnames
    ]

    problem.x_guesses_full = pstart[pnames].values

    return petab_problem, problem


if __name__ == '__main__':
    MODEL_NAME = sys.argv[1]

    load_problem(MODEL_NAME, force_compile=True)
