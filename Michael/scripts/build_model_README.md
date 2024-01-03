# build_model instructions

This script takes as input the name and the savepath of the model and builds the model that corresponds to the name. There are 4 different models in the script.

You can see an example of a batch script to launch this program (my_job_build_model.sh) with some example parameters. You can launch that script with the command "sbatch my_job_build_model.sh".

# Parameters

--savepath_model:                 Path in which you want to save the model
--model_name:                     name of the model, choose betwenn the following ones:
                                      'CNN_base_0':     simple CNN for 2.5 degree data 
                                      'Unet_base_0':    simple Unet for 2.5 degree data 
                                      'CNN_1_0':        simple CNN for 1 degree data 
                                      'Unet_1_0':       simple Unet for 1 degree data 
