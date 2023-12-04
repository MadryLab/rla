"""Parse the results of all dTERMen runs ran in a directory and create a summary csv.

Usage:
    .. code-block::

        python summarize_results.py \\
            --output_dir <folder_containing_dTERMen_runs> \\
            --dtermen_data <dtermen_data_root>

See :code:`python summarize_results.py --help` for more info.
"""
import argparse
import glob
import os
import traceback

import pandas as pd

alphabet = ['D', 'E', 'K', 'R', 'H', 'Q', 'N', 'S', 'T', 'P', 'G', 'A', 'V', 'I', 'L', 'M', 'C', 'F', 'W', 'Y', 'X']

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse all results.')
    parser.add_argument('--output_dir', help='Output directory', required=True)
    parser.add_argument('--dtermen_data', help="Root directory for dTERMen runs", required=True)
    args = parser.parse_args()

    output_path = os.path.join(args.output_dir, 'etabs')
    errors = []

    ids = []
    pred_sequences = []
    real_sequences = []
    dtermen_pred_sequences = []
    recovery = []
    energy = []
    energy_u = []  # mean energy
    energy_s = []  # energy std
    dtermen_recovery = []
    dtermen_energy = []
    dtermen_energy_u = []
    dtermen_energy_s = []
    for filename in glob.glob(os.path.join(output_path, '*-output.out')):
        pdb_id = os.path.basename(filename)[:-len('-output.out')]
        ids += [pdb_id]
        print(pdb_id)
        try:
            with open(filename, 'r') as f:
                f.readline()
                f.readline()
                p_seq, p_nrg = f.readline().split('|')[0:2]
                r_seq, r_nrg = f.readline().split('|')[0:2]
                pred_sequences += [p_seq]
                real_sequences += [r_seq]
                p_nrg = float(p_nrg.strip())
                energy.append(p_nrg)
                # TODO: compute nrg of native sequence given etab if not provided
                r_nrg = r_nrg.strip()
                if r_nrg == 'N/A':
                    pass

                recov = float(f.readline().split('|')[0][:-2])

                # if recovery is 0, double check that it's actually 0
                # by computing the recovery from the given sequences
                if recov == 0.0:
                    length = 0
                    for i in range(len(r_seq)):
                        if r_seq[i] in alphabet:
                            length += 1
                            if p_seq[i] == r_seq[i]:
                                recov += 1
                    recov = recov / length * 100
                recovery += [recov]

                nrg_u = float(f.readline().split(' ')[-1])
                energy_u.append(nrg_u)
                nrg_s = float(f.readline().split(' ')[-1])
                energy_s.append(nrg_s)

        except Exception as e:
            traceback.print_exc()
            errors.append(pdb_id)
            continue

        last_testfolder = None
        for testfolder in glob.glob(args.dtermen_data):
            already_found = False
            already_found_recov = False
            already_found_mean = False
            already_found_std = False
            if os.path.isdir(os.path.join(testfolder, pdb_id)):
                last_testfolder = testfolder
                if os.path.isfile(os.path.join(testfolder, pdb_id, 'design.oFIXED')):
                    fixed_file = os.path.join(testfolder, pdb_id, 'design.oFIXED')
                    with open(fixed_file, 'r') as f:
                        for line in f:
                            linesplit = line.split('|')
                            if len(linesplit) >= 3:
                                if linesplit[2] == ' lowest-energy sequence\n' and not already_found:
                                    nrg = float(linesplit[1].strip())
                                    dtermen_energy += [nrg]
                                    dtermen_pred_sequences += [linesplit[0]]
                                    already_found = True
                                elif len(linesplit) >= 4 and linesplit[3] == ' recovery\n' and not already_found_recov:
                                    dtermen_recovery += [float(linesplit[0][:-2])]
                                    already_found_recov = True
                            elif len(linesplit) == 1:
                                linesplit = line.split(' ')
                                if linesplit[0] == 'mean' and not already_found_mean:
                                    nrg_u = float(linesplit[-1].strip())
                                    dtermen_energy_u.append(nrg_u)
                                    already_found_mean = True
                                elif linesplit[0] == 'estimated' and not already_found_std:
                                    nrg_s = float(linesplit[-1].strip())
                                    dtermen_energy_s.append(nrg_s)
                                    already_found_std = True
                                    break

                for dtermen_filename in glob.glob(os.path.join(testfolder, pdb_id, 'design.o*')):
                    with open(dtermen_filename, 'r') as f:
                        for line in f:
                            linesplit = line.split('|')
                            if len(linesplit) >= 3:
                                if linesplit[2] == ' lowest-energy sequence\n' and not already_found:
                                    nrg = float(linesplit[1].strip())
                                    dtermen_energy += [nrg]
                                    dtermen_pred_sequences += [linesplit[0]]
                                    already_found = True
                                if len(linesplit) >= 4 and linesplit[3] == ' recovery\n' and not already_found_recov:
                                    dtermen_recovery += [float(linesplit[0][:-2])]
                                    already_found_recov = True
                            elif len(linesplit) == 1:
                                linesplit = line.split(' ')
                                if linesplit[0].strip() == 'mean' and not already_found_mean:
                                    nrg_u = float(linesplit[-1].strip())
                                    dtermen_energy_u.append(nrg_u)
                                    already_found_mean = True
                                elif linesplit[0].strip() == 'estimated' and not already_found_std:
                                    nrg_s = float(linesplit[-1].strip())
                                    dtermen_energy_s.append(nrg_s)
                                    already_found_std = True
                                    break
            if already_found and already_found_recov and already_found_mean and already_found_std:
                break

        num_seq = len(pred_sequences)

        # pad the lists if the corresponding dTERMen values aren't available
        if len(dtermen_pred_sequences) < num_seq:
            dtermen_pred_sequences += ['UNKNOWN']
        if len(dtermen_recovery) < num_seq:
            dtermen_recovery += [-1]
            with open('to_run.out', 'a') as f:
                f.write(os.path.join(last_testfolder, pdb_id) + '\n')
        if len(dtermen_energy) < num_seq:
            dtermen_energy += [None]
        if len(dtermen_energy_u) < num_seq:
            dtermen_energy_u += [None]
        if len(dtermen_energy_s) < num_seq:
            dtermen_energy_s += [None]

    os.chdir(output_path)
    for etab_file in glob.glob("*.etab"):
        pdb = etab_file[:-5]
        if not os.path.exists(f"{pdb}-output.out") and pdb not in errors:
            errors.append(pdb)

    print("Errors:", errors)
    for e in errors:
        sbatch_file = os.path.join(output_path, f"run_{e}.sh")
        os.system(f"sbatch {sbatch_file}")

    results_dict = {
        'ids': ids,
        'pred_sequences': pred_sequences,
        'real_sequences': real_sequences,
        'dtermen_pred_sequences': dtermen_pred_sequences,
        'recovery': recovery,
        'dtermen_recovery': dtermen_recovery,
        'energy': energy,
        'energy_mean': energy_u,
        'energy_std': energy_s,
        'dtermen_energy': dtermen_energy,
        'dtermen_energy_mean': dtermen_energy_u,
        'dtermen_energy_std': dtermen_energy_s,
    }
    print([(key, len(val)) for key, val in results_dict.items()])
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(os.path.join(args.output_dir, 'summary_results.csv'))
