# nc_to_csv instructions

This script takes as input the sourcepaths of the grib files togheter with the savepaths and some other parameters and produces the corresponding csv files. It works both with 2.5 degree and 1 degree data.

You can see an example of a batch script to launch this program (my_job_create_csv.sh) with some example parameters. You can launch that script with the command "sbatch my_job_create_csv.sh". This script is built to be parallelizable, the example batch is set to make it run on 3 CPUs.


# Parameters

### Paths parameters

The following parameters are paths used for reading grib files and writing csv files. Using default paths you will read the 2.5 grib files from a folder in the shared location of the CLINT repository and you will write the csv files in another folder in the same location. Remember to change all the paths to switch from 2.5 to 1 (in the my_job_create_csv.sh file you see an example of the parameters used to create csv of 1 degree).

--savepath_features:            path to save all the features in a unique csv file
--savepath_features_train:      path to save training features (until 2010-12-31)
--savepath_features_val:        path to save validation features (since 2011-01-01 until 2015-12-31)
--savepath_features_test:       path to save test features (since 2016-04-01 2022-12-31)
--savepath_target:              path to save all the targets in a unique csv file
--savepath_target_train:        path to save training targets (until 2010-12-31)
--savepath_target_val:          path to save validation targets (since 2011-01-01 until 2015-12-31)
--savepath_target_test:         path to save test targets (since 2016-04-01 2022-12-31)
--sourcepath_features:          source path of the features (directory that contains grib files)
--sourcepath_target:            source path of the targets (directory that contains grib files)

### Other parameters

Use the following parameters to choose which rectangle to use for extracting features to put in csv files. Use min year and max year to extract only features of specific years. Default parameters refer to Southern Indian with years range 1980-2022.

--min_lat
--max_lat
--min_lon
--max_lon
--min_year
--max_year

The following parameters are used only in particular cases, you can ignore them and use the default values.

--crossing
--min_lon2
--min_lat2