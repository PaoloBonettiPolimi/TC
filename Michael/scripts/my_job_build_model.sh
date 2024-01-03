#!/usr/bin/env bash

#SBATCH -J modeljob
#SBATCH --output my_log_build_model.log
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH -A bk1318
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --gpus=1
module load clint tf
python build_model.py --model_name "CNN_1_0" --savepath_model "/work/bk1318/b382633/models/fresh_models/CNN_1_0"