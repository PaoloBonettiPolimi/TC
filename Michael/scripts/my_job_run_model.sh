#!/usr/bin/env bash

#SBATCH -J CNNjob
#SBATCH --output my_log_run_model.log
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH -A bk1318
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --gpus=1
module load clint tf
python run_model.py --save "yes" --data_path "/work/bk1318/b382633/data_npy/npy_1/" --sourcepath_model "/work/bk1318/b382633/models/fresh_models/CNN_1_0/" --savepath_model "/work/bk1318/b382633/models/trained_models/CNN_1_0/" --savepath_prediction "/work/bk1318/b382633/predictions/CNN_1_0/"