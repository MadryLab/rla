"""Split dataset into multiple training folds"""
import argparse
import glob
import os
import random

INPUT_DATA = '/pool001/users/vsundar/TERMinator/'
MIN_PROT_LEN = 30


def main(args):
    dataset_files = os.path.join(INPUT_DATA, args.dataset)
    num_folds = args.folds
    pdb_ids = []
    for filename in glob.glob(os.path.join(dataset_files, '*', '*.features')):
        prefix = os.path.splitext(filename)[0]
        with open(f'{prefix}.length') as fp:
            fp.readline()
            seq_len = int(fp.readline().strip())
            if seq_len < MIN_PROT_LEN:
                continue
            pdb_ids += [prefix[-4:]]

    random.shuffle(pdb_ids)

    out_folder = os.path.join(dataset_files, args.outfolder)
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    len_fold = int(len(pdb_ids) / num_folds)
    folds = []
    for i in range(num_folds):
        if i == num_folds - 1:
            endval = len(pdb_ids)
        else:
            endval = (i + 1) * len_fold
        folds += [pdb_ids[int(i * len_fold):endval]]
        with open(os.path.join(out_folder, f'fold_{i}.in'), 'w') as f:
            for pdb_id in folds[i]:
                f.write(pdb_id + '\n')

    with open(os.path.join(out_folder, 'holdout_fold.in'), 'w') as f:
        for pdb_id in folds[-1]:
            f.write(pdb_id + '\n')

    for i in range(num_folds - 1):
        test_fold = i
        val_fold = i - 1
        if val_fold == -1:
            val_fold += num_folds - 1
        training_folds = list(set(range(num_folds - 1)) - set([test_fold, val_fold]))
        print(training_folds, val_fold, test_fold)
        with open(os.path.join(out_folder, f'train_fold{i}.in'), 'w') as f:
            for fold in training_folds:
                for pdb_id in folds[fold]:
                    f.write(pdb_id + '\n')
        with open(os.path.join(out_folder, f'val_fold{i}.in'), 'w') as f:
            for pdb_id in folds[val_fold]:
                f.write(pdb_id + '\n')
        with open(os.path.join(out_folder, f'test_fold{i}.in'), 'w') as f:
            for pdb_id in folds[test_fold]:
                f.write(pdb_id + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Split Data for TERMinator')
    parser.add_argument('--dataset',
                        help='input folder .features files in proper directory structure. prefix is $ifsdata/',
                        default='features_singlechain')
    parser.add_argument('--folds', help='number of folds', default='11', type=int)
    parser.add_argument('--outfolder', help='folder to store fold splits in', default='fold_splits')
    args = parser.parse_args()
    main(args)
