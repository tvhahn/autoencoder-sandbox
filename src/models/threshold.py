import numpy as np
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import random
import seaborn as sns

import tensorflow as tf
# from tensorflow import keras

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils import shuffle

class SelectThreshold:
    
    def __init__(self, model, X_val, y_val, X_val_slim, class_to_remove, class_normal, class_names, model_name):
        
        self.model = model
        self.X_val = X_val
        self.y_val = y_val
        self.X_val_slim = X_val_slim
        self.class_to_remove = class_to_remove
        self.class_normal = class_normal
        self.class_names = class_names
        self.model_name = model_name
        
        print(np.shape(self.X_val))
        
        # build the reconstructions on the X_val_slim dataset, and the X_val dataset
        self.recon_val_slim = self.model(self.X_val_slim).numpy()
        self.recon_val = self.model(self.X_val).numpy()
        
    def mse(self, X_val, recon_val):
        """Calculate MSE for images in X_val and recon_val"""
        # need to calculate mean across the rows, and then across the columns
        return np.mean(np.mean(np.square(X_val - recon_val),axis=1),axis=1)

    def rmse(self, X_val, recon_val):
        """Calculate RMSE for images in X_val and recon_val"""
        return np.sqrt(self.mse(X_val, recon_val))

    def euclidean_distance(self, X_val, recon_val):
        dist = np.linalg.norm(X_val - recon_val,axis=(1,2))
        return dist
    
    # function that creates a pandas dataframe with the RMSE value, and the associated class
    def create_df_reconstruction(self, reconstruction_error_val, threshold_val):
        df = pd.DataFrame(data=reconstruction_error_val, columns=["metric"])

        class_names_list = list(zip(self.class_names, range(len(self.class_names))))
        
        y_names = []
        for i in self.y_val:
            y_names.append(str(i)+", "+class_names_list[i][0])
        
        # append the class values
        df['class'] = self.y_val
        df['class_names'] = y_names

        # label anomolous (outlier) data as -1, inliers as 1
            # -1 (outlier) is POSITIVE class
            #  1 (inlier) is NEGATIVE class
        new_y_val = []
        for i in self.y_val:
            if i in self.class_to_remove:
                new_y_val.append(-1)
            else:
                new_y_val.append(1)

        df['true_class'] = new_y_val

        # add prediction based on threshold
        df['prediction'] = np.where(df['metric'] >= threshold_val,-1,1)

        return df
    
    def threshold_grid_search(self, lower_bound, upper_bound, reconstruction_error_val, grid_iterations=10):
        '''Simple grid search for finding the best threshold'''
    
        roc_scores = {}
        grid_search_count = 0
        for i in np.arange(lower_bound, upper_bound, (np.abs(upper_bound-lower_bound) / grid_iterations)):
#             if grid_search_count%50 == 0:
#                 print('grid search iteration: ', grid_search_count)

            threshold_val = i
            df = self.create_df_reconstruction(reconstruction_error_val, 
                                          threshold_val)
            roc_val = roc_auc_score(df['true_class'], df['prediction'])
            roc_scores[i] = roc_val
            grid_search_count += 1

        # return best roc_score and the threshold used to set it
        threshold_val = max(zip(roc_scores.values(), roc_scores.keys()))
        best_threshold = threshold_val[1]
        best_roc_score = threshold_val[0]
        print('Best threshold value:', best_threshold,'\tROC score: {:.2%}'.format(best_roc_score))

        # use the best threshold value to make a confusion matrix
        df = self.create_df_reconstruction(reconstruction_error_val, best_threshold)
        
        return df, best_threshold, best_roc_score
    
    def box_plot(self, df, best_threshold, best_roc_score, metric):
        fig, ax = plt.subplots(figsize=(12,5))
        df.boxplot(column=['metric'], by='class_names', ax=ax).axhline(y=best_threshold,c='red',alpha=0.7)
        plt.title('Boxplots of {} for X_valid, by Class'.format(metric))
        plt.suptitle('')
        plt.show()
        
        print('\nConfusion Matrix:')
        print(confusion_matrix(df['true_class'], df['prediction']))
        
    # function to test the different reconstruction methods (mse, rmse, euclidean)
    # do a grid search looking for the best threshold, and then outputting the results
    def compare_error_method(self,show_results=True, grid_iterations=10):
        '''Function to test the different reconstruction methods (mse, rmse, euclidean) 

        Parameters
        ===========
        model : tensorflow model
            autoencoder model that was trained on the "slim" data set.
            Will be used to build reconstructions

        X_val : ndarray
            tensor of the X validation set

        class_to_remove : ndarray
            numpy array of the classes to remove from the X_val and y_val data
        '''
        
        col = ['model_name','class_normal','method','best_threshold','best_roc_score']
        result_table = pd.DataFrame(columns=col)
        
        # build the reconstructions on the X_val_slim dataset, and the X_val dataset
        recon_val_slim = self.model(self.X_val_slim).numpy()
        recon_val = self.model(self.X_val).numpy()

        # run through each of the reconstruction error methods, perform a little grid search
        # to find the optimum value

        #_______MSE_______#
        # calculate MSE reconstruction error
        mse_recon_val_slim = self.mse(self.X_val_slim, recon_val_slim) # for slim dataset
        mse_recon_val = self.mse(self.X_val, recon_val) # for complete validation dataset
        
        max_mse = np.max(mse_recon_val_slim)
        percentile_mse = np.percentile(mse_recon_val_slim,90)

        lower_bound = percentile_mse - percentile_mse*0.9
        upper_bound = max_mse*1.5

        df, best_threshold, best_roc_score = self.threshold_grid_search(lower_bound,
                                                                                upper_bound,
                                                                                mse_recon_val,grid_iterations)
        result_table = result_table.append(pd.DataFrame([[self.model_name, self.class_normal,'mse',
                                                          best_threshold, 
                                                          best_roc_score]],
                                                        columns=col))
        
        if show_results == True:
            self.box_plot(df, best_threshold, best_roc_score, 'MSE')
        
        #_______RMSE_______#
        # calculate RMSE reconstruction error
        rmse_recon_val_slim = self.rmse(self.X_val_slim, recon_val_slim) # for slim dataset
        rmse_recon_val = self.rmse(self.X_val, recon_val) # for complete validation dataset
        
        max_rmse = np.max(rmse_recon_val_slim)
        percentile_rmse = np.percentile(rmse_recon_val_slim,90)

        lower_bound = percentile_rmse - percentile_rmse*0.9
        upper_bound = max_rmse*1.5

        df, best_threshold, best_roc_score = self.threshold_grid_search(lower_bound,
                                                                                upper_bound,
                                                                                rmse_recon_val,grid_iterations)
        
        result_table = result_table.append(pd.DataFrame([[self.model_name, self.class_normal,'rmse',
                                                  best_threshold, 
                                                  best_roc_score]],
                                                columns=col))
        
        if show_results == True:
            self.box_plot(df, best_threshold, best_roc_score, 'RMSE')
        
        #_______Euclidean_______#
        # calculate Euclidean reconstruction error
        eu_recon_val_slim = self.euclidean_distance(self.X_val_slim, recon_val_slim) # for slim dataset
        eu_recon_val = self.euclidean_distance(self.X_val, recon_val) # for complete validation dataset
        
        max_eu = np.max(eu_recon_val_slim)
        percentile_eu = np.percentile(eu_recon_val_slim,90)

        lower_bound = percentile_eu - percentile_eu*0.9
        upper_bound = max_eu*1.5

        df, best_threshold, best_roc_score = self.threshold_grid_search(lower_bound,
                                                                      upper_bound, 
                                                                      eu_recon_val,grid_iterations)
        
        result_table = result_table.append(pd.DataFrame([[self.model_name, self.class_normal,'euclid_dist',
                                          best_threshold, 
                                          best_roc_score]],
                                        columns=col))
        if show_results == True:
            self.box_plot(df, best_threshold, best_roc_score, 'Euclidean Distance')
        
        return result_table