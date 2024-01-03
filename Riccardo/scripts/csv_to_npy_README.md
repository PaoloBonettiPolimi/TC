# csv_to_npy instructions

This script takes as input the sourcepaths of the csv files togheter with the savepaths and some other parameters and produces the corresponding npy files. It works both with 2.5 degree and 1 degree data.

You can see an example of a batch script to launch this program (my_job_npy.sh) with some example parameters. You can launch this script with the command "sbatch my_job_npy.sh".

# Parameters

### Paths parameters:

The following parameters are paths used for reading csv files and writing npy files. Using default paths you will read the 2.5 csv files from a folder in the shared location of the CLINT repository and you will write the npy files in another folder in the same location. Remember to change all the paths to switch from 2.5 to 1 (in the my_job_npy.sh file you see an example of the parameters used with 1 degree data).

--datapath:                     path of the directory that contains the csv files to read.
--savepath:                     path of the directory you want to save numpy files in

IMPORTANT: files in datapath have to be named:
'train_dailymeans_'+area+'.csv'
'val_dailymeans_'+area+'.csv'
'test_dailymeans_'+area+'.csv'
'target_train_dailymeans_'+area+'.csv'
'target_val_dailymeans_'+area+'.csv'
'target_test_dailymeans_'+area+'.csv'

### Other parameters:

--area                          the name of the area (default 'Sindian')
--single_std:                   'yes' if you want to standardize the data (recommended)
--features_list:                list of features to include (default is 'vo', 'r', 'u_200', 'u_850',                                                         'v_200', 'v_850', 'sst', 'tcwv', 'tclw', 'tciw', 'shear')