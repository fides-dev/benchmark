import pypesto.petab
import petab
import os
import sys

folder_base = os.path.join(os.path.dirname(__file__),
                           'Benchmark-Models-PEtab',
                           'Benchmark-Models')


def preprocess_problem(problem, model):
    if model == 'Weber_BMC2015':
        # don't estimate certain standard deviation parameters
        problem.parameter_df.loc[
            ['std_yPKDt', 'std_yPI4K3Bt', 'std_yCERTt'],
            petab.ESTIMATE
        ] = 0

    if model in ['Brannmark_JBC2010', 'Fiedler_BMC2016']:
        petab.flatten_timepoint_specific_output_overrides(problem)


def load_problem(model, force_compile=False):
    yaml_config = os.path.join(folder_base, model, model + '.yaml')
    petab_problem = petab.Problem.from_yaml(yaml_config)
    preprocess_problem(petab_problem, model)
    importer = pypesto.petab.PetabImporter(petab_problem)
    problem = importer.create_problem(force_compile=force_compile)
    return petab_problem, problem


if __name__ == '__main__':
    MODEL_NAME = sys.argv[1]

    load_problem(MODEL_NAME, force_compile=True)
