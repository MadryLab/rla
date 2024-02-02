This is the associated code for ["Jointly Embedding Protein Structures and Sequences through Residue Level Alignment"](https://www.mlsb.io/papers_2023/Jointly_Embedding_Protein_Structures_and_Sequences_through_Residue_Level_Alignment.pdf) by Foster Birnbaum, Saachi Jain, Aleksander Madry, and Amy E. Keating.

## Setup 

### Requirements
The [rla_env.yml](rla_env.yml) file specifies the needed requirements.

### Model weights and data

Model weights and data are available to download [here](https://www.dropbox.com/scl/fo/97tmmxudd9yftv5p4fywm/h?rlkey=7ezbmiorplryxmy42gjuojz34&dl=0). The model weights folder should be downloaded to the home directory. The data are provided as a zipped folder containing the train/validation/test datasplits as WebDatasets.

Example inference data are provided in the [example_data](example_data) folder. 

## Inference

Once the model weights folder is downloaded, the value of the `model_dir = /c/example/path` line in each example notebook must be changed to reflect the path to the weights. Additionally, if the computer you are running the notebook on is offline, the `args_dict['arch'] = '/c/example/path'` line must be changed to reflect the path to ESM-2 as downloaded by the [transformers](https://huggingface.co/docs/transformers/index) module. If your computer is online, the `args_dict['arch'] = '/c/example/path'` should be deleted.

### Structural candidate ranking

An example of how to use RLA to rank candidate structures is provided [here](example_notebooks/example_decoy.ipynb). The example ranks hundreds of decoy structures for 2 real structures from the PDB and evaluates the comparison by calculating a correlation to the decoy TM-scores. The data are sourced from [Roney and Ovchinnikov, 2022](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.129.238101).

### Mutation effect prediction

An example of how to use RLA to predict the effect of mutations is provided [here](example_notebooks/example_mutation.ipynb). The example predicts the effects of thousands of single and double amino acid substitutions on the stability of single chain proteins and compares the predictions to experimentally observed values. The data are sourced from [Tsuboyama et al., 2023](https://www.nature.com/articles/s41586-023-06328-6).

### Contact prediction

An example of how to use RLA to predict the contacts between 2 residues in a protein is provided [here](example_notebooks/example_contact.ipynb).

## Training

Details on how to train RLA are coming soon.
