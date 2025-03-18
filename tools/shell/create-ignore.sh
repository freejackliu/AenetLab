#!/bin/bash
#SBATCH -p *?* -n *?* -J *?*

create_set --ignore
find . -type f -name "core*" -delete
