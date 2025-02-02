import pypesto.petab
import petab
import os
import sys

import pandas as pd
import numpy as np

PARAMETER_ALIASES = {
    'Fujita_SciSignal2010': {
        'init_AKT': 'init_Akt',
        'scaling_pAkt_tot': 'scaleFactor_pAkt',
        'scaling_pEGFR_tot': 'scaleFactor_pEGFR',
        'scaling_pS6_tot': 'scaleFactor_pS6'
    },
    'Zheng_PNAS2012': {
        'sigma': 'noise'
    },
    'Bruno_JExpBot2016': {
        'init_b10_1': 'init_b10',
        'init_bcry_1': 'init_bcry',
        'init_zea_1': 'init_zea',
        'init_ohb10_1': 'init_ohb10'
    },
    'Isensee_JCB2018': {
        'rho_pRII_Western': 'sigma_pRII_Western',
        'rho_Calpha_Microscopy': 'sigma_Calpha',
        'rho_pRII_Microscopy': 'sigma_pRII'
    }
}

MODEL_ALIASES = {
    'Crauste_CellSystems2017': 'Crauste_ImmuneCells_CellSystems2017',
    'Bruno_JExpBot2016': 'Bruno_Carotines_JExpBio2016',
    'Schwen_PONE2014': 'Schwen_InsulinMouseHepatocytes_PlosOne2014',
    'Beer_MolBioSystems2014': 'Beer_MolBiosyst2014',
    'Lucarelli_CellSystems2018': 'Lucarelli_TGFb_2017'
}

folder_base = os.path.join(os.path.dirname(__file__),
                           'Benchmark-Models-PEtab',
                           'Benchmark-Models')


def preprocess_problem(problem, model, extend_bounds):
    if model in ['Brannmark_JBC2010', 'Fiedler_BMC2016']:
        petab.flatten_timepoint_specific_output_overrides(problem)

    if np.isfinite(extend_bounds):
        problem.parameter_df[petab.LOWER_BOUND] /= extend_bounds
        problem.parameter_df[petab.UPPER_BOUND] *= extend_bounds
    else:
        problem.parameter_df.loc[
            problem.parameter_df[petab.PARAMETER_SCALE] == petab.LIN,
            petab.LOWER_BOUND
        ] = - np.inf
        problem.parameter_df.loc[
            problem.parameter_df[petab.PARAMETER_SCALE] != petab.LIN,
            petab.LOWER_BOUND
        ] = 0
        problem.parameter_df[petab.UPPER_BOUND] = np.inf


def load_problem(model, force_compile=False, extend_bounds=1.0):
    yaml_config = os.path.join(folder_base, model, model + '.yaml')
    petab_problem = petab.Problem.from_yaml(yaml_config)
    preprocess_problem(petab_problem, model, extend_bounds)
    importer = pypesto.petab.PetabImporter(petab_problem, validate_petab=False)
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
