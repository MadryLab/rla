#!/bin/bash

. /etc/profile.d/modules.sh

mkdir /home/gridsan/mlu/keatinglab_shared/mlu/tuning/include_nlcpl/${2}
for i in 0 1 2
do
	for j in 0.01 0.1 1 10 100 1000
	do
		sed -e "s/SORTCERY/$j/g" </home/gridsan/mlu/refactor_TERMinator/hparams/run/finetune_sortcery_noreg.json >/home/gridsan/mlu/refactor_TERMinator/hparams/run/finetune_sortcery_noreg_${j}.json
		cp -r /home/gridsan/mlu/keatinglab_shared/mlu/tuning/templates/triplicate/${1}${i}/. /home/gridsan/mlu/keatinglab_shared/mlu/tuning/include_nlcpl/${2}/run${i}_${j}/
		. /home/gridsan/mlu/refactor_TERMinator/scripts/models/train/submit_train.sh /home/gridsan/mlu/keatinglab_shared/mlu/bcl2/term_features_with_sortcery /home/gridsan/mlu/refactor_TERMinator/hparams/model/ablate_singleton_features.json /home/gridsan/mlu/refactor_TERMinator/hparams/run/finetune_sortcery_noreg_${j}.json /home/gridsan/mlu/keatinglab_shared/mlu/tuning/include_nlcpl/${2}/run${i}_${j} /home/gridsan/mlu/keatinglab_shared/mlu/tuning/include_nlcpl/${2}/run${i}_${j} 12 train validation test
	done
done
