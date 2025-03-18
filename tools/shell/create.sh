#!/bin/bash
#SBATCH -p *?* -n *?* -J *?*

create_set
find . -type f -name "core*" -delete
