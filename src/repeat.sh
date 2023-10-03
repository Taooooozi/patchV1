#!/bin/bash

cfg_fdr=/home/ljy/repos/patchV1/src 
repeats=9 # 重复次数-1
minimalTC=minimalTC_repeat
minimal_cfg=minimal_repeat.cfg

cd ${cfg_fdr}
# echo ${cfg_fdr}/${minimalTC}
${cfg_fdr}/${minimalTC}
for repeat in $( seq 1 $repeats )
do
    vim -s ${cfg_fdr}/nextpatch_minimalTC.keys ${cfg_fdr}/${minimalTC}
    vim -s ${cfg_fdr}/nextpatch_minimalcfg.keys ${cfg_fdr}/${minimal_cfg}
    ${cfg_fdr}/${minimalTC}
done