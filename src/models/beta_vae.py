import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
import datetime

# build custom sampling function
K = keras.backend

class Sampling(keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return K.random_normal(tf.shape(log_var)) * K.exp(log_var / 2) + mean

# rounded accuracy for the metric
def rounded_accuracy(y_true, y_pred):
    return keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))

def model_fit(X_train_slim, X_val_slim, beta_value=0.25, codings_size=10, seed=42, epochs=500, earlystop_patience=8, verbose=0):

    tf.random.set_seed(seed)
    np.random.seed(seed)


    inputs = keras.layers.Input(shape=[28, 28])
    z = keras.layers.Flatten()(inputs)
    z = keras.layers.Dense(150, activation="selu")(z)
    z = keras.layers.Dense(100, activation="selu")(z)
    codings_mean = keras.layers.Dense(codings_size)(z)
    codings_log_var = keras.layers.Dense(codings_size)(z)
    codings = Sampling()([codings_mean, codings_log_var])
    variational_encoder_beta = keras.models.Model(
        inputs=[inputs], outputs=[codings_mean, codings_log_var, codings])

    decoder_inputs = keras.layers.Input(shape=[codings_size])
    x = keras.layers.Dense(100, activation="selu")(decoder_inputs)
    x = keras.layers.Dense(150, activation="selu")(x)
    x = keras.layers.Dense(28 * 28, activation="sigmoid")(x)
    outputs = keras.layers.Reshape([28, 28])(x)
    variational_decoder_beta = keras.models.Model(inputs=[decoder_inputs], outputs=[outputs])

    _, _, codings = variational_encoder_beta(inputs)
    reconstructions = variational_decoder_beta(codings)
    variational_ae_beta = keras.models.Model(inputs=[inputs], outputs=[reconstructions])

    latent_loss = -0.5 * beta_value * K.sum(
        1 + codings_log_var - K.exp(codings_log_var) - K.square(codings_mean),
        axis=-1)
    variational_ae_beta.add_loss(K.mean(latent_loss) / 784.)
    variational_ae_beta.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=[rounded_accuracy])

    # use tensorboard to track training
    log_dir="logs/" +str('vae_')+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, 
                                                          histogram_freq=0,
                                                          update_freq='epoch',
                                                          profile_batch=0)

    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=earlystop_patience, 
                                                          restore_best_weights=True)

    history = variational_ae_beta.fit(X_train_slim, X_train_slim, epochs=epochs,
                               validation_data=[X_val_slim, X_val_slim], 
                               callbacks=[tensorboard_callback,earlystop_callback],verbose=verbose)
    return variational_ae_beta