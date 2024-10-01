Simple instructions for starting up RLA:
    1. Use rla.yml to create the proper environment # TODO: Update the .yml file to include all needed packages, as it is missing a few small ones that currently need to be installed manually
    2. Use write_WDS.ipynb (or a .py equivalent if you want to create one) to convert from .pdb files into a .wds format readable by RLA
    3. Use run_RLA.ipynb to calculate RLA scores for each .pdb structure and analyze the results