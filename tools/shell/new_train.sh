#!/bin/bash
#SBATCH -p test -n 24 -J debug

parallel_prepare
replica_number=5
epoch_number=100
for ((i=1;i<=replica_number;i++))
do
    parallel_train ${i} ${epoch_number} --earlystop
done
parallel_buildpool
find . -type f -name "core*" -delete
