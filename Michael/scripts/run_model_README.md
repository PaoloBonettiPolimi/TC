# run_model instructions

This script takes as input the sourcepath of the npy files and the sourcepath of the model togheter with the savepaths of trained models and predictions and some other parameters. It trains the input model using the input data and saves the predictions and the trained model in the selected folders. It works both with 2.5 degree and 1 degree data.

You can see an example of a batch script to launch this program (my_job_run_model.sh) with some example parameters. You can launch that script with the command "sbatch my_job_run_model.sh".

# Parameters:

### Path parameters

The following parameters are paths used for reading the npy files and the model and writing the csv file of predictions and the resulting models. Using default paths you will read the 2.5 npy files from a folder in the shared location of the repository and you will write the csv file in another folder in the same location. Remember to change all the paths to switch from 2.5 to 1 (in the my_job_run_model.sh file you see an example of the parameters used with 1 degree data).

--datapath:                     path of the directory that contains the npy files to read
--sourcepath_model:             path of the model to train
--savepath_model:               path in which you want to save the trained models
--savepath_prediction:          path in which you want to save the predictions
--x_test_path:                  since the predictions will be saved in a csv file that contains the features 
                                of the test data in the first columns and the predicted probablities in the last 
                                column, you have to provide the path of the csv file that contains train features 
                                of 2.5 degree data, both for 2.5 and 1 degree predictions (remember that targets 
                                and predictions will always have 2.5 granularity).
                                
IMPORTANT: files in datapath have to be named:
area+'_train_img_std.npy'
area+'_val_img_std.npy'
area+'_test_img_std.npy'
area+'_y_train_img.npy'
area+'_y_val_img.npy'
area+'_y_test_img.npy'

### Other parameters:

--area:                         the name of the area (default 'Sindian')
--save:                         'yes' if you want to save the predictions
--columns:                      the indices of the columns of the npy files you want to use. It has to be a unique 
                                string with comma-separated indices. Default is "0,1,2,3,4,5,6,7,10".