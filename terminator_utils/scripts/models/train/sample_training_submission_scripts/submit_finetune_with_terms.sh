#!/bin/bash

. /etc/profile.d/modules.sh

mkdir /home/gridsan/mlu/keatinglab_shared/mlu/tuning/triplicate/${2}
for i in 0 1 2
do
	for j in 3e-5
	do
		sed -e "s/LR/$j/g" </home/gridsan/mlu/refactor_TERMinator/hparams/run/finetune_sortcery_only.json >/home/gridsan/mlu/refactor_TERMinator/hparams/run/finetune_sortcery_only_${i}_${j}.json
		cp -r /home/gridsan/mlu/keatinglab_shared/mlu/tuning/templates/triplicate/${1}${i}/. /home/gridsan/mlu/keatinglab_shared/mlu/tuning/triplicate/${2}/run${i}_${j}/
		. /home/gridsan/mlu/refactor_TERMinator/scripts/models/train/submit_train.sh /home/gridsan/mlu/keatinglab_shared/mlu/bcl2/term_features_with_sortcery /home/gridsan/mlu/refactor_TERMinator/hparams/model/ablate_singleton_features.json /home/gridsan/mlu/refactor_TERMinator/hparams/run/finetune_sortcery_only_${i}_${j}.json /home/gridsan/mlu/keatinglab_shared/mlu/tuning/triplicate/${2}/run${i}_${j} /home/gridsan/mlu/keatinglab_shared/mlu/tuning/triplicate/${2}/run${i}_${j} 12 train validation test
	done
done
