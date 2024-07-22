from change_chain_id import rename_pred_chains, rename_native_chains
import csv
from subprocess import getstatusoutput 
import sys
import os
import warnings
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)
warnings.filterwarnings('ignore')

'''
This script runs DockQ, including preprocessing steps such as renaming the chains in the
predicted & native structures and aligning the residue numbering.

It expects change_chain_id.py to be in the same directory (see import statements).
Set global variables RENUMBERING_PATH and DOCKQ_PATH for DockQ executables.

Usage: run_dockq.py pred.pdb native.pdb
'''

RENUMBERING_PATH = '/data1/groups/keatinglab/DockQ/scripts/fix_numbering.pl'
DOCKQ_PATH = '/data1/groups/keatinglab/DockQ/DockQ.py'

def run_dockq(pred_path, native_path, output_path=None):
    renamed_pred_path = rename_pred_chains(pred_path)
    renamed_native_path = rename_pred_chains(native_path)

    s1, output1 = getstatusoutput(f'{RENUMBERING_PATH} {renamed_pred_path} {renamed_native_path}')
	
    clean_pred_path = renamed_pred_path + '.fixed'
    status, output = getstatusoutput(f'python {DOCKQ_PATH} {clean_pred_path} {renamed_native_path} -short')
    output_dict = {}
    if status == 0:
        output_dict['model'] = clean_pred_path
        output_dict['native'] = renamed_native_path
        output_dict['DockQ'] = output.split('DockQ')[-1].strip().split(' ')[0]
        output_dict['Fnat'] = output.split('Fnat')[-1].strip().split(' ')[0]
        output_dict['iRMS'] = output.split('iRMS')[-1].strip().split(' ')[0]
        output_dict['LRMS'] = output.split('LRMS')[-1].strip().split(' ')[0]
        output_dict['Fnonnat'] = output.split('Fnonnat')[-1].strip().split(' ')[0]
        
        
        if not output_path:
            model = pred_path.split('.')[0]    
            output_path = model + '_dockq.csv'
        else:
            model = os.path.basename(pred_path).split('.')[0]
            output_path = os.path.join(output_path, model + '_dockq.csv')
        with open(output_path,'w') as f:
            w = csv.writer(f)
            w.writerow(output_dict.keys())
            w.writerow(output_dict.values())


if __name__ == "__main__":
    run_dockq(sys.argv[1], sys.argv[2])