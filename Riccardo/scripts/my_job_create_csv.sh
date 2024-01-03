#!/usr/bin/env bash

#SBATCH -J my_job_create_csv_final
#SBATCH --output my_log_create_csv.log
#SBATCH -A bk1318
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=32G

source activate xarray_env
python nc_to_csv_final.py --savepath_features '/work/bk1318/b382633/data_csv/predictors_v2_dailymean_1/features_dailymeans_Sindian.csv' --savepath_features_train '/work/bk1318/b382633/data_csv/predictors_v2_dailymean_1/train_dailymeans_Sindian.csv' --savepath_features_val '/work/bk1318/b382633/data_csv/predictors_v2_dailymean_1/val_dailymeans_Sindian.csv' --savepath_features_test '/work/bk1318/b382633/data_csv/predictors_v2_dailymean_1/test_dailymeans_Sindian.csv' --savepath_target '/work/bk1318/b382633/data_csv/predictors_v2_dailymean_1/target_dailymeans_Sindian.csv' --savepath_target_train '/work/bk1318/b382633/data_csv/predictors_v2_dailymean_1/target_train_dailymeans_Sindian.csv' --savepath_target_val '/work/bk1318/b382633/data_csv/predictors_v2_dailymean_1/target_val_dailymeans_Sindian.csv' --savepath_target_test '/work/bk1318/b382633/data_csv/predictors_v2_dailymean_1/target_test_dailymeans_Sindian.csv' --sourcepath_features '/work/bk1318/b382633/data_grib/predictors_v2_dailymean_1' --sourcepath_target '/work/bk1318/b382633/data_grib/target'