"""Patch dTERMen runs with errors."""
import argparse
import glob
import os
import sys

# for autosummary import purposes
sys.path.insert(0, os.path.dirname(__file__))
from search_utils import find_pdb_path

DIR = os.path.dirname(os.path.abspath(__file__))
assert DIR[0] == "/", "DIR should be an abspath"

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Patch dTERMen runs with errors.')
    parser.add_argument('--output_dir', help="Location to place patch runs")
    parser.add_argument('--pdb_root', help="Root directory for dTERMen runs")
    args = parser.parse_args()

    with open('to_run.out', 'r') as f:
        for line in f:
            filepath = line[:-1]
            pdb_id = filepath[-4:]
            print(pdb_id)
            os.system(f'cp -r {filepath} {args.output_dir}')
            out_dir = os.path.join(args.output_dir, pdb_id)
            pdb_path = find_pdb_path(pdb_id, args.pdb_root)
            source_pdb_file = os.path.join(pdb_path, pdb_id.lower()[1:3], f"{pdb_id}.pdb")
            out_pdb_file = os.path.join(out_dir, f"{pdb_id}.pdb")
            os.system(f'cp {source_pdb_file} {out_pdb_file}')
            run_script_path = os.path.join(out_dir, f"fix_{pdb_id}.sh")
            os.system((f"sed -e \"s|ID|{pdb_id}|g\" "
                       f"-e \"s|OUTPUTDIR|{args.output_dir}|g\" "
                       f"<{os.path.join(DIR, 'fix_dTERMen.sh')} "
                       f">{run_script_path}"))
            os.system(f"cd {os.path.join(args.output_dir, pdb_id)} && sbatch fix_{pdb_id}.sh")
