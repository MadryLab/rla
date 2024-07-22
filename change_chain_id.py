from Bio.PDB import PDBList, PDBIO, PDBParser, is_aa, Model
import sys


'''
These methods change the chain IDs of the protein-peptide complex to be "A" and "B" (for DockQ compatibility)
'''

def rename_pred_chains(pdb_path):
    root_path = '/'.join(pdb_path.split('/')[0:-1])
    pdb = pdb_path.split('/')[-1].split('.')[0]

    pdb_io = PDBIO()
    pdb_struct = PDBParser().get_structure(pdb, pdb_path)

    chain_1, chain_2 = pdb_struct.get_chains()
    chain_1_length = len([r for r in chain_1.get_residues() if is_aa(r)])
    chain_2_length = len([r for r in chain_2.get_residues() if is_aa(r)])
    if chain_1_length > chain_2_length:
        try:
            chain_1.id = 'A'
            chain_2.id = 'B'
        except:
            chain_2.id = 'Z'
            chain_1.id = 'A'
            chain_2.id = 'B'
            
    else:
        try:
            chain_1.id = 'B'
            chain_2.id = 'A'
        except:
            chain_2.id = 'Z'
            chain_1.id = 'B'
            chain_2.id = 'A'

    pdb_io.set_structure(pdb_struct)
    pdb_io.save(f'{root_path}/{pdb}_chain_renamed.pdb')
    return f'{root_path}/{pdb}_chain_renamed.pdb'

def get_all_chains(structure):
    yield from structure.get_chains()

def rename_native_chains(pdb_path):
    root_path = '/'.join(pdb_path.split('/')[0:-1])
    pdb = pdb_path.split('/')[-1].split('.')[0]


    pdb_io = PDBIO()
    pdb_struct = PDBParser().get_structure(pdb, pdb_path)


    chain_1_id = root_path[-2]
    chain_2_id = root_path[-1]

    chain_1 = None
    chain_2 = None

    for chain in list(get_all_chains(pdb_struct)):
        if chain.id != chain_1_id and chain.id != chain_2_id:
            chain.get_parent().detach_child(chain.id)
        elif chain.id == chain_1_id:
            chain_1 = chain
        elif chain.id == chain_2_id:
            chain_2 = chain

    chain_2.id = 'tmp'
    chain_1.id = 'A'
    chain_2.id = 'B'

    pdb_io.set_structure(pdb_struct)
    pdb_io.save(f'{root_path}/{pdb}_chain_renamed.pdb')
    return f'{root_path}/{pdb}_chain_renamed.pdb'


if __name__ == "__main__":
    if 'model' in sys.argv[1]:
        rename_pred_chains(sys.argv[1])
    else:
        rename_native_chains(sys.argv[1])
