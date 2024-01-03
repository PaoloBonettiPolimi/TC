#!/usr/bin/env bash

#SBATCH -J unetIndices
#SBATCH --output unetIndices95.log
#SBATCH -p gpu
#SBATCH -A bk1318
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --gpus=1

module load clint tf
python cnn_multipleRegions_server.py --save='yes' --savepath_prediction='/work/bk1318/b382633/predictions/unet_indices_flattening_95PCA_latAndLon_' --data_path='/home/b/b382633/TC_adds/indices/' --single_std='yes'