#!/bin/bash
#SBATCH -p *?* -n *?* -J *?*

taylor_train
find . -type f -name "core*" -delete
