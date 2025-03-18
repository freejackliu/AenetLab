#!/bin/bash
#SBATCH -p *?* -n *?* -J *?*

cleanse -E 0 -F 3
find . -type f -name "core*" -delete
