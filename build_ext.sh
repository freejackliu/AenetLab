#!/bin/bash

PATH_SPLIT=`pwd | awk -F"python/" '{print $2}'`
AENET_HOME=`pwd | awk -F"python/" '{print $1}'` 
if [ "$PATH_SPLIT" == "aenet/AenetLab" ]; then 
    cp ext/* ../
    ln -s ${AENET_HOME}bin/generate.x*serial ${AENET_HOME}bin/generate.x
    ln -s ${AENET_HOME}bin/train.x*mpi ${AENET_HOME}bin/train.x
    echo "Build ext scripts successfully!"
else
    echo "Setup failed! Move 'AenetLab' src directory to 'path_to_aenet/python/aenet'."
fi
