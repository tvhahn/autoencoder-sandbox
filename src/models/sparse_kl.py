import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import datetime

def rounded_accuracy(y_true, y_pred):
    return keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))

# build custom regularizer for KL-Divergence
K = keras.backend
kl_divergence = keras.losses.kullback_leibler_divergence

class KLDivergenceRegularizer(keras.regularizers.Regularizer):
    def __init__(self, weight, target=0.1):
        self.weight = weight
        self.target = target
    def __call__(self, inputs):
        mean_activities = K.mean(inputs, axis=0)
        return self.weight * (
            kl_divergence(self.target, mean_activities) +
            kl_divergence(1. - self.target, 1. - mean_activities))
    
def rounded_accuracy(y_true, y_pred):
    return keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))

def model_fit(X_train_slim, X_val_slim, seed=42, epochs=500, earlystop_patience=8, verbose=0):

    tf.random.set_seed(seed)
    np.random.seed(seed)

    kld_reg = KLDivergenceRegularizer(weight=0.05, target=0.1)
    sparse_kl_encoder = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(100, activation="selu"),
        keras.layers.Dense(300, activation="sigmoid", activity_regularizer=kld_reg)
    ])
    sparse_kl_decoder = keras.models.Sequential([
        keras.layers.Dense(100, activation="selu", input_shape=[300]),
        keras.layers.Dense(28 * 28, activation="sigmoid"),
        keras.layers.Reshape([28, 28])
    ])
    sparse_kl_ae = keras.models.Sequential([sparse_kl_encoder, sparse_kl_decoder])
    sparse_kl_ae.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(),
                  metrics=[rounded_accuracy])

    # show summary, if wanted
    # sparse_kl_encoder.summary()
    # sparse_kl_decoder.summary()

    # use tensorboard to track training
    log_dir="logs/"+"skl_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, 
                                                          histogram_freq=0,
                                                          update_freq='epoch',
                                                          profile_batch=0)

    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=earlystop_patience, 
                                                          restore_best_weights=True)

    history = sparse_kl_ae.fit(X_train_slim, X_train_slim, epochs=epochs,
                               validation_data=[X_val_slim, X_val_slim], 
                               callbacks=[tensorboard_callback,earlystop_callback],verbose=verbose)
    return sparse_kl_ae