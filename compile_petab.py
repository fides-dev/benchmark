import pypesto.petab
import petab
import os
import sys
import re
import pandas as pd


def fix_fiedler(petab_problem):
    petab_problem.parameter_df = petab_problem.parameter_df.append(
        pd.Series({
            petab.PARAMETER_NAME: 'sigma_{pErk}',
            petab.PARAMETER_SCALE: petab.LOG10,
            petab.LOWER_BOUND: 1e-5,
            petab.UPPER_BOUND: 1e3,
            petab.NOMINAL_VALUE: 0.04527013133744955,
            petab.ESTIMATE: 1.0
        }, name='sigma_pErk'))
    petab_problem.parameter_df = petab_problem.parameter_df.append(
        pd.Series({
            petab.PARAMETER_NAME: 'sigma_{pMek}',
            petab.PARAMETER_SCALE: petab.LOG10,
            petab.LOWER_BOUND: 1e-5,
            petab.UPPER_BOUND: 1e3,
            petab.NOMINAL_VALUE: 0.0005804511382145272,
            petab.ESTIMATE: 1.0
        }, name='sigma_pMek'))
    petab_problem.parameter_df.drop(index=[
        'pErk_20140430_gel1_sigma', 'pMek_20140430_gel1_sigma',
        'pErk_20140505_gel1_sigma', 'pMek_20140505_gel1_sigma',
        'pErk_20140430_gel2_sigma', 'pMek_20140430_gel2_sigma',
        'pErk_20140505_gel2_sigma', 'pMek_20140505_gel2_sigma',
    ], inplace=True)
    new_measurement_dfs = []
    new_observable_dfs = []
    for (obs_id, noise_par, obs_par), measurements in \
            petab_problem.measurement_df.groupby([
                petab.OBSERVABLE_ID, petab.NOISE_PARAMETERS,
                petab.OBSERVABLE_PARAMETERS
            ]):
        replacement_id = f'{obs_id}_{obs_par[-4:]}'

        measurements.drop(columns=[petab.NOISE_PARAMETERS,
                                   petab.OBSERVABLE_PARAMETERS],
                          inplace=True)
        measurements[petab.OBSERVABLE_ID] = replacement_id

        observable = petab_problem.observable_df.loc[obs_id].copy()
        observable.name = replacement_id
        for target in [petab.OBSERVABLE_FORMULA,
                       petab.NOISE_FORMULA]:
            observable[target] = re.sub(
                fr'observableParameter[0-9]+_{obs_id}',
                obs_par,
                observable[petab.OBSERVABLE_FORMULA]
            )
        observable[petab.NOISE_FORMULA] = re.sub(
            r'^pERK',
            r'sigma_pErk',
            observable[petab.OBSERVABLE_FORMULA]
        )
        observable[petab.NOISE_FORMULA] = re.sub(
            r'^pMEK',
            r'sigma_pMek',
            observable[petab.NOISE_FORMULA]
        )
        new_measurement_dfs.append(measurements)
        new_observable_dfs.append(observable)

    petab_problem.observable_df = pd.concat(new_observable_dfs, axis=1).T
    petab_problem.observable_df.index.name = petab.OBSERVABLE_ID
    petab_problem.measurement_df = pd.concat(new_measurement_dfs)


folder_base = os.path.join(os.path.dirname(__file__),
                           'Benchmark-Models-PEtab',
                           'Benchmark-Models')


def preprocess_problem(problem, model):
    if model == 'Chen_MSB2009':
        # don't estimate parameters on linear scale
        problem.parameter_df.loc[
            problem.parameter_df[petab.PARAMETER_SCALE] == petab.LIN,
            petab.ESTIMATE
        ] = 0

    if model == 'Weber_BMC2015':
        # don't estimate parameters on linear scale
        problem.parameter_df.loc[
            ['std_yPKDt', 'std_yPI4K3Bt', 'std_yCERTt'],
            petab.ESTIMATE
        ] = 0

    if model == 'Fiedler_BMC2016':
        fix_fiedler(problem)

    if model == 'Brannmark_JBC2010':
        petab.flatten_timepoint_specific_output_overrides(problem)


def load_problem(model, force_compile=False):
    yaml_config = os.path.join(folder_base, model, model + '.yaml')
    petab_problem = petab.Problem.from_yaml(yaml_config)
    preprocess_problem(petab_problem, model)
    importer = pypesto.petab.PetabImporter(petab_problem)
    return petab_problem, importer.create_problem(force_compile=force_compile)


if __name__ == '__main__':
    MODEL_NAME = sys.argv[1]

    load_problem(MODEL_NAME, force_compile=True)
