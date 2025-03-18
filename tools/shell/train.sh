#!/bin/bash
#SBATCH -p *?*
#SBATCH -N *?*
#SBATCH -n *?*
#SBATCH -J *?*

parallel "parallel-train {}" ::: 0 1 2 3 4 5 6 7 8 9
find . -type f -name "core*" -delete
