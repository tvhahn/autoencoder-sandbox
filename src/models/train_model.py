import numpy as np
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import random
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
import tensorboard

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.utils import shuffle

print('TensorFlow version: ', tf.__version__)
print('Keras version: ', keras.__version__)
print('Tensorboard version:', tensorboard.__version__)

from src.data import data_prep
from src.models import sparse_kl, beta_vae, threshold

# create dataframe to store all the results
col = ['model_name','class_normal','method','best_threshold','best_roc_score']
df_all = pd.DataFrame(columns=col)

# fashion MNIST data
fashion_mnist = keras.datasets.fashion_mnist

run_no = 1

for class_normal in range(6,7):
    for i in range(0,5):
        ### DATA PREP ####
        print('Run no: ', run_no, ' Class: ', class_normal)

        class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                       "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
        
        random_int = random.randint(0,500)
        print('Random seed: ',random_int)
        
        data_model = data_prep.DataPrep(fashion_mnist,class_normal, random_int=random_int)

        (X_train, y_train, 
         X_train_slim, y_train_slim,
         X_val, y_val,
         X_val_slim, y_val_slim,
         X_test,y_test) = data_model.train_test_split()

        class_to_remove = [i for i in range(0,10)]
        class_to_remove.remove(class_normal)
        class_to_remove = np.array(class_to_remove,dtype='uint8')

        ### TRY MODELS ###

        # SPARSE-KL
        # sparse_kl_ae = sparse_kl.model_fit(X_train_slim, X_val_slim, seed=random_int, epochs=500, earlystop_patience=8, verbose=0)
        # recon_check = threshold.SelectThreshold(sparse_kl_ae, X_val, y_val, X_val_slim, class_to_remove, class_normal, class_names, "sparse_kl")
        # df = recon_check.compare_error_method(show_results=False, grid_iterations=100)
        # df_all = df_all.append(df)

        # BETA-VAE
        beta_vae_model = beta_vae.model_fit(X_train_slim, X_val_slim, beta_value=10, codings_size=10, seed=random_int, epochs=500, earlystop_patience=250, verbose=0)
        recon_check = threshold.SelectThreshold(beta_vae_model, X_val, y_val, X_val_slim, class_to_remove, class_normal, class_names, "beta-vae_beta=10_early-stop=120")
        df = recon_check.compare_error_method(show_results=False, grid_iterations=100)
        df_all = df_all.append(df)

        # BETA-VAE
        beta_vae_model = beta_vae.model_fit(X_train_slim, X_val_slim, beta_value=5, codings_size=5, seed=random_int, epochs=500, earlystop_patience=120, verbose=0)
        recon_check = threshold.SelectThreshold(beta_vae_model, X_val, y_val, X_val_slim, class_to_remove, class_normal, class_names, "beta-vae_beta=5_early-stop=120_size=5")
        df = recon_check.compare_error_method(show_results=False, grid_iterations=100)
        df_all = df_all.append(df)

        # BETA-VAE
        beta_vae_model = beta_vae.model_fit(X_train_slim, X_val_slim, beta_value=5, codings_size=15, seed=random_int, epochs=500, earlystop_patience=120, verbose=0)
        recon_check = threshold.SelectThreshold(beta_vae_model, X_val, y_val, X_val_slim, class_to_remove, class_normal, class_names, "beta-vae_beta=5_early-stop=120_size=15")
        df = recon_check.compare_error_method(show_results=False, grid_iterations=100)
        df_all = df_all.append(df)

        
        run_no += 1

print(df_all.head())

# save the dataframe of the results
df_all.to_csv('results_2.csv')