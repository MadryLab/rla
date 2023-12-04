"""Convert .pdb files into protein backbone .red.pdb files.

Usage:
    .. code-block::

        python cleanStructs.py \\
            --in_list_path <pdb_paths_file> \\
            --out_folder <output_folder> \\
            [-n <num_processes>]

    :code:`<pdb_paths_file>` should be a file of paths to .pdb files, with one path per line

    :code:`<output_folder>` will be where the outputted .red.pdb files are dumped, and will
    be structured as :code:`<output_folder>/<pdb_id>/<pdb_id>.red.pdb`

See :code:`python cleanStructs.py --help` for more info.
"""
import argparse
import multiprocessing as mp
import os
import sys
import traceback

import numpy as np

# pylint: disable=unspecified-encoding


def extractBackbone(filename, outpath):
    """Given a PDB structure, extract the protein backbone atoms and dump it in a redesigned PDB file.

    Args
    ----
    filename : str
        Input .pdb file
    outpath : str
        Prefix to place the output file (.red.pdb will be appended)
    """
    VALID_ELEMENTS = ['N', 'CA', 'C', 'O']
    VALID_RECORD_TYPES = ['ATOM', 'HETATM']
    struct_dict = {}
    valid_entry_lines = []
    with open(filename, 'r') as fp:
        entry_lines = [l for l in fp]

    for line_num, line in enumerate(entry_lines):
        data = line.strip()
        if data[:3] == 'TER' or data[:3] == 'END':
            valid_entry_lines.append(line_num)
            continue
        record_type = data[0:6].strip()
        if record_type not in VALID_RECORD_TYPES:
            print(f"Skipping line: {data}")
            continue

        try:
            element = data[13:16].strip()
            residx = data[22:27].strip()
            chain = data[21]
        except Exception as e:
            print(data)
            raise e


        if (chain, residx) not in struct_dict.keys():
            struct_dict[(chain, residx)] = {"elements": np.array([False for _ in range(5)]), "line_numbers": []}

        if element in VALID_ELEMENTS:
            struct_dict[(chain, residx)]["elements"][VALID_ELEMENTS.index(element)] = True
            struct_dict[(chain, residx)]["line_numbers"].append(line_num)
        elif element == 'OXT':
            struct_dict[(chain, residx)]["elements"][-1] = True
            struct_dict[(chain, residx)]["oxt_num"] = line_num

    for struct_vals in struct_dict.values():
        elem_arr = struct_vals["elements"]
        if elem_arr[:4].all():
            # if we have N, CA, C, O, we take those lines
            # and ignore OXT even if present
            valid_entry_lines += struct_vals["line_numbers"]
        elif elem_arr[[0, 1, 2, 4]].all() and not elem_arr[3]:
            # if we have N, CA, C, OXT, but no O
            # we take OXT as O
            assert len(struct_vals["line_numbers"]) == 3, struct_vals["line_numbers"]
            valid_entry_lines += struct_vals["line_numbers"]
            valid_entry_lines.append(struct_vals["oxt_num"])

    valid_entry_lines.sort()

    with open(outpath, 'w') as fp:
        for idx, _ in enumerate(valid_entry_lines):
            cur_line_num = valid_entry_lines[idx]
            prev_line_num = valid_entry_lines[idx - 1] if idx > 0 else valid_entry_lines[0]
            cur_line = entry_lines[cur_line_num]
            prev_line = entry_lines[prev_line_num]
            if prev_line.strip() == 'TER' and cur_line.strip() == 'TER':
                # prevent redundant TER if we filter out a whole section
                continue
            fp.write(cur_line)


def _raise_error(error):
    """Wrapper for error handling without crashing"""
    traceback.print_exception(Exception, error, None)


# inner loop we want to parallize
def dataGen(in_path, out_folder):
    """Wrapper for :code:`extractBackbone` for path manipuation and error catching.

    Args
    ----
    in_path : str
        input .pdb to :code:`extractBackbone`
    out_folder : str
        output directory to dump .red.pdb files into
    """
    name = os.path.basename(in_path)[:-len(".pdb")]
    data_folder = os.path.join(out_folder, name)
    if not os.path.isdir(data_folder):
        os.mkdir(data_folder)
    out_file = os.path.join(out_folder, name, f"{name}.red.pdb")
    print('out file', out_file)
    try:
        extractBackbone(in_path, out_file)
        assert os.path.exists(out_file)
    except Exception as e:
        print(out_file, file=sys.stderr)
        raise e


# when subprocesses fail you usually don't get an error...
def generateCoordsDir(in_list, out_folder, num_cores=1):
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
    # make folder where the dataset files are gonna be placed
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    # generate absolute paths so i dont have to think about relative references
    out_folder = os.path.abspath(out_folder)

    pool = mp.Pool(num_cores, maxtasksperchild=10)
    for in_file in in_list:
        in_file = os.path.abspath(in_file)
        pool.apply_async(dataGen, args=(in_file, out_folder), error_callback=_raise_error)

    pool.close()
    pool.join()
    print("Done")


if __name__ == '__main__':
    # idek how to do real parallelism but this should fix the bug of stalling when processes crash
    mp.set_start_method("spawn")  # i should use context managers but low priority change
    parser = argparse.ArgumentParser('Extract backbone from a list of PDB files')
    parser.add_argument('--in_list_path',
                        help='file that contains paths to PDB files to clean, with one path per line.',
                        required=True)
    parser.add_argument('--out_folder',
                        help=('folder where cleaned .red.pdb files will be placed. '
                              'folder organization is <out_folder>/<pdb_id>/<pdb_id>.red.pdb'),
                        required=True)
    parser.add_argument('-n', dest='num_cores', help='number of cores to use', default=1, type=int)
    args = parser.parse_args()
    with open(args.in_list_path) as fp:
        in_list = [l.strip() for l in fp]

    generateCoordsDir(in_list, args.out_folder, num_cores=args.num_cores)
