import pypesto.petab
import petab
import os
import sys

folder_base = os.path.join(os.path.dirname(__file__),
                           'Benchmark-Models-PEtab',
                           'Benchmark-Models')

if __name__ == '__main__':
    MODEL_NAME = sys.argv[1]

    yaml_config = os.path.join(folder_base, MODEL_NAME, MODEL_NAME + '.yaml')

    petab_problem = petab.Problem.from_yaml(yaml_config)
    importer = pypesto.petab.PetabImporter(petab_problem)
    problem = importer.create_model(force_compile=True)
