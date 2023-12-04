#!/bin/bash

. /etc/profile.d/modules.sh

for i in 3 10 30 100
do
	for j in 1e-5 3e-5
	do
		sed -e "s/SORTCERY/$i/g" -e "s/LR/$j/g" </home/gridsan/mlu/refactor_TERMinator/hparams/run/finetune_sortcery_noreg.json >/home/gridsan/mlu/refactor_TERMinator/hparams/run/finetune_sortcery_noreg_${i}_${j}.json
		cp -r /home/gridsan/mlu/keatinglab_shared/mlu/TERMinator_runs/tune/template_noreg/. /home/gridsan/mlu/keatinglab_shared/mlu/TERMinator_runs/tune/run_noreg/multi_combined_${i}_${j}/
		. /home/gridsan/mlu/refactor_TERMinator/scripts/models/train/submit_train.sh /home/gridsan/mlu/keatinglab_shared/mlu/bcl2/term_features_with_sortcery /home/gridsan/mlu/refactor_TERMinator/hparams/model/delete_net1.json /home/gridsan/mlu/refactor_TERMinator/hparams/run/finetune_sortcery_noreg_${i}_${j}.json /home/gridsan/mlu/keatinglab_shared/mlu/TERMinator_runs/tune/run_noreg/multi_combined_${i}_${j} /home/gridsan/mlu/keatinglab_shared/mlu/TERMinator_runs/tune/run_noreg/multi_combined_${i}_${j} 10 train validation test
	done
done
