"""Generate feature files for TERMinator.

Usage:
    .. code-block::

        python generateDataset.py \\
            --in_folder <input_folder> \\
            --out_folder <output_folder> \\
            [--cutoff <matches_cutoff>] \\
            [-n <num_processes>] \\
            [-u] \\ # update existing files
            [--coords_only] \\
            [--dummy_terms [None, 'replace', 'include']]

    :code:`--in_folder <input_folder>` should be structured as :code:`<input_folder>/<pdb_id>/<pdb_id>.<ext>`.
    For full feature generation, :code:`ext` must include :code:`.dat` and :code:`.red.pdb`, while
    if running using :code:`--coords_only` only :code:`.red.pdb` is required.
    If you use :code:`scripts/data/preprocessing/cleanStructs.py`, this structure is automatically built.

    :code:`--out_folder <output_folder>` will be structured as :code:`<input_folder>/<pdb_id>/<pdb_id>.<ext>`,
    where :code:`<ext>` includes :code:`.features`, which specifies protein and TERM features, and
    :code:`.length`, which contains two integerss. The first integer specifies the number of TERM residues
    in the protein, while the second integer specifies the sequence length of the protein.

    :code:`--cutoff <matches_cutoff>` restricts the number of matches featurized to the top :code:`<matches_cutoff>`,
    ranked by increasing RMSD. Defaults to 50.

    :code:`-n <num_processes>` specifies how many processes to use while processing. Defaults to 1.

    :code:`[-u]` is an optional flag which, if specified, forces rewriting of existing feature files.

    :code:`--coords_only` is an option flag which, if specified, generated only backbone-derived features.
    Running this mode does not require prior TERM mining, but does require you clean the backbone using
    :code:`scripts/data/preprocessing/cleanStructs.py`.

    :code:`--dummy_terms` allows specifying how dummy TERMs are incorperated into features. Dummy TERMs are
    constructs where there is one TERM match with a degenerate X sequence and structural features derived from
    the target structure, By default, it is set to :code:`None`, or no dummy TERMs. If set to :code:`'replace'`,
    only the dummy TERM is included. If set to :code:`'include'`, the first match is set to the dummy TERM match
    and the remaining TERMs are those parsed from the :code:`.dat` file.

See :code:`python generateDataset.py --help` for more info.
"""
import argparse
import functools
import glob
import multiprocessing as mp
import os
import sys
import traceback

# for autosummary import purposes
sys.path.insert(0, os.path.dirname(__file__))
from packageTensors import dumpCoordsTensors, dumpTrainingTensors


# when subprocesses fail you usually don't get an error...
def generateDatasetParallel(in_folder,
                            out_folder,
                            cutoff=50,
                            num_cores=1,
                            update=True,
                            coords_only=False,
                            dummy_terms=None):
    """Parallelize :code:`dataGen` over a list of files.

    Args
    ----
    in_folder : str
        Path to input directory in proper structure
    out_folder : str
        Path to the output folder
    cutoff : int
        Max number of TERMs to featurize
    num_cores : int
        Number of processes to parallelize with
    update : bool
        Whether or not to overwrite existing files
    coords_only : bool
        Whether to use only backbone-derived features
    dummy_terms : str or None
        Method by which to incorperate dummy TERMs. Options include :code:`'replace'`,
        which means replacing TERM features with those derived from a dummy TERM, or
        :code:`'include'`, which includes the dummy TERM into the mined TERM matches.
    """
    print('num cores', num_cores)
    print(('warning! it seems that if subprocesses fail right now you don\'t get an error message. '
           'be wary of this if the number of files you\'re getting seems off'))
    # make folder where the dataset files are gonna be placed
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    # generate absolute paths so i dont have to think about relative references
    in_folder = os.path.abspath(in_folder)
    out_folder = os.path.abspath(out_folder)

    os.chdir(in_folder)

    process_func = functools.partial(dataGen, cutoff=cutoff, coords_only=coords_only, dummy_terms=dummy_terms)

    pool = mp.Pool(num_cores, maxtasksperchild=10)
    # process folder by folder
    for folder in glob.glob("*"):
        # folders that aren't directories aren't folders!
        if not os.path.isdir(folder):
            continue

        full_folder_path = os.path.join(out_folder, folder)
        if not os.path.exists(full_folder_path):
            os.mkdir(full_folder_path)

        for _, file in enumerate(glob.glob(folder + '/*.red.pdb')):
            name = file[:-len(".red.pdb")]
            if not update:
                out_file = os.path.join(out_folder, name)
                if os.path.exists(out_file + '.features'):
                    continue

            pool.apply_async(process_func, args=(file, out_folder), error_callback=_raise_error)

    pool.close()
    pool.join()


def _raise_error(error):
    """Wrapper for error handling without crashing"""
    traceback.print_exception(Exception, error, None)


# inner loop we wanna parallize
def dataGen(file, out_folder, cutoff, coords_only, dummy_terms):
    """Wrapper function for parallelization which deals with paths and other args.

    Args
    ----
    file : str
        The .red.pdb file for the protein to featurize.
    out_folder : str
        Path to the output folder
    cutoff : int
        Max number of TERMs to featurize
    coords_only : bool
        Whether to use only backbone-derived features
    dummy_terms : str or None
        Method by which to incorperate dummy TERMs. Options include :code:`'replace'`,
        which means replacing TERM features with those derived from a dummy TERM, or
        :code:`'include'`, which includes the dummy TERM into the mined TERM matches.
    """
    name = file[:-len(".red.pdb")]
    out_file = os.path.join(out_folder, name)
    print('out file', out_file)
    try:
        if coords_only:
            dumpCoordsTensors(name, out_path=out_file)
        else:
            dumpTrainingTensors(name,
                                out_path=out_file,
                                cutoff=cutoff,
                                coords_only=coords_only,
                                dummy_terms=dummy_terms)
    except Exception as e:
        print(out_file, file=sys.stderr)
        raise e


if __name__ == '__main__':
    # idek how to do real parallelism but this should fix the bug of stalling when processes crash
    mp.set_start_method("spawn")  # i should use context managers but low priority change
    parser = argparse.ArgumentParser('Generate features data files from dTERMen .dat files')
    parser.add_argument('--in_folder',
                        help='input folder containing .dat/.red.pdb files in proper directory structure',
                        required=True)
    parser.add_argument('--out_folder', help='folder where features will be placed', required=True)
    parser.add_argument('--cutoff', dest='cutoff', help='max number of match entries per TERM', default=50, type=int)
    parser.add_argument('-n', dest='num_cores', help='number of processes to use', default=1, type=int)
    parser.add_argument('-u',
                        dest='update',
                        help='if added, update existing files. else, files that already exist will not be overwritten',
                        default=False,
                        action='store_true')
    parser.add_argument('--coords_only',
                        dest='coords_only',
                        help='if added, only include coordinates-relevant data in the feature files',
                        default=False,
                        action='store_true')
    parser.add_argument('--dummy_terms', help='option for how to use dummy TERMs in the feature files', default=None)
    args = parser.parse_args()
    generateDatasetParallel(args.in_folder,
                            args.out_folder,
                            cutoff=args.cutoff,
                            num_cores=args.num_cores,
                            update=args.update,
                            coords_only=args.coords_only,
                            dummy_terms=args.dummy_terms)
