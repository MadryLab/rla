"""Convert dTERMen .etab files to numpy arrays of Potts model parameters.

Usage:
    .. code-block::

        python dtermen2npEtabs.py \\
            --in_list_path <etab_paths_file> \\
            --out_folder <output_folder> \\
            -n <num_processes>

    :code:`<etab_paths_file>` should be a file of paths to .etab files, with one path per line

See :code:`python dtermen2npEtabs.py --help` for more info.
"""
import argparse
import glob
import multiprocessing as mp
import os
import sys
import traceback

import re
import numpy as np

# Code from preprocessing folder
sys.path.append("../")
from data.preprocessing.parseEtab import parseEtab


def parseEtabs(out_folder, in_list, num_cores=1):
    """Parallelize :code:`dataGen` over a list of files.

    Args
    ----
    in_list : list of paths
        List of input paths to :code:`dataGen`.
    out_folder : str
        Path to the output folder
    """
    print('num cores', num_cores)
    print(('warning! it seems that if subprocesses fail right now you don\'t get an error message. '
           'be wary of this if the number of files you\'re getting seems off'))
    out_folder = os.path.abspath(out_folder)

    pool = mp.Pool(num_cores, maxtasksperchild=10)

    for filepath in in_list:
        res = pool.apply_async(dataGen, args=(filepath, out_folder), error_callback=_raise_error)

    pool.close()
    pool.join()


def _raise_error(error):
    """Wrapper for error handling without crashing"""
    traceback.print_exception(Exception, error, None)


# inner loop we wanna parallize
def dataGen(filepath, out_folder):
    """Wrapper for :code:`parseEtab` for path manipuation and error catching.

    Args
    ----
    in_path : str
        input .etab to :code:`parseEtab`
    out_folder : str
        output directory to dump .etab.npy files into
    """
    name = re.split('/|\\\\', filepath)[-1]
    out_file = os.path.join(out_folder, name)
    print('out file', out_file)
    try:
        _, _, etab = parseEtab(filepath, save=False)
        np.save(out_file, etab)
    except Exception as e:
        print(out_file, file=sys.stderr)
        print(e)
        raise e


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse dTERMen etabs and dump numpy versions')
#    parser.add_argument('--in_list_path',
#                        help='file containing etab paths, with one etab path per line',
#                        required=True)
    parser.add_argument('--out_folder', help='folder where numpy etabs will be placed', required=True)
    parser.add_argument('-n', dest='num_cores', help='number of proceses to use', type=int, default=1)
    args = parser.parse_args()

#    with open(args.in_list_path) as fp:
#        in_list = [l.strip() for l in fp]
    
    in_files = glob.glob(args.out_folder+"/*.etab")
    in_list = [os.path.abspath(i) for i in in_files]

    parseEtabs(args.out_folder, in_list, num_cores=args.num_cores)
