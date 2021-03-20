import pypesto.petab
import petab
import os
import sys

from fix_fiedler import fix_fiedler

folder_base = os.path.join(os.path.dirname(__file__),
                           'Benchmark-Models-PEtab',
                           'Benchmark-Models')

if __name__ == '__main__':
    MODEL_NAME = sys.argv[1]

    yaml_config = os.path.join(folder_base, MODEL_NAME, MODEL_NAME + '.yaml')

    petab_problem = petab.Problem.from_yaml(yaml_config)
    if MODEL_NAME == 'Fiedler_BMC2016':
        fix_fiedler(petab_problem)
    if MODEL_NAME == 'Brannmark_JBC2010':
        petab.flatten_timepoint_specific_output_overrides(petab_problem)
    importer = pypesto.petab.PetabImporter(petab_problem)
    problem = importer.create_model(force_compile=True)
