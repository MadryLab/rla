#!/bin/bash

. /etc/profile.d/modules.sh

for i in 0.5 1 2 4
do
	sed -e "s/REGULARIZATION/$i/g" </home/gridsan/mlu/refactor_TERMinator/hparams/run/copy_ingraham_run.json >/home/gridsan/mlu/refactor_TERMinator/hparams/run/copy_ingraham_run_${i}.json
	. /home/gridsan/mlu/refactor_TERMinator/scripts/models/train/submit_train.sh /home/gridsan/mlu/keatinglab_shared/alexjli/TERMinator/features/features_ingraham_coords_only /home/gridsan/mlu/refactor_TERMinator/hparams/model/copy_ingraham.json /home/gridsan/mlu/refactor_TERMinator/hparams/run/copy_ingraham_run_${i}.json /home/gridsan/mlu/keatinglab_shared/mlu/TERMinator_runs/singlechain_reg_${i} /home/gridsan/mlu/keatinglab_shared/mlu/TERMinator_runs/singlechain_reg_${i} 72
done
