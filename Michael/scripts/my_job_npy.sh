#!/usr/bin/env bash

#SBATCH -J my_job_npy
#SBATCH --output my_log_npy.log
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH -A bk1318
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --gpus=1
module load clint tf
python csv_to_npy_final.py --savepath "/work/bk1318/b382633/data_npy/npy_1/" --single_std='yes' --data_path="/work/bk1318/b382633/data_csv/predictors_v2_dailymean_1/" --area "Sindian"