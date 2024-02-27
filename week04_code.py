''' 
Main reference: github/ageron/handson-ml2/blob/master/11_training_deep_neural_networks.ipynb
Last review: March 2022
'''

# In[0]: IMPORTS AND COMMON SETTINGS
#region
import sys
assert sys.version_info >= (3, 5) # Python ≥3.5 is required
import tensorflow as tf
assert tf.__version__ >= "2.0" # TensorFlow ≥2.0 is required
from tensorflow import keras
from tensorflow.keras import callbacks, layers, metrics, regularizers
from tensorflow.python.keras import activations
from tensorflow.python.keras.backend import softmax
from tensorflow.python.keras.layers.advanced_activations import ELU, LeakyReLU, ReLU
from tensorflow.python.keras.layers.core import Dropout
import sklearn
assert sklearn.__version__ >= "0.20" # Scikit-Learn ≥0.20 is required
import numpy as np
import os
import joblib
import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
np.random.seed(42) # to make the output stable across runs
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
# To run multiple instance of VS code
#physical_devices = tf.config.list_physical_devices('GPU') # to run multiple instance of VS code
#tf.config.experimental.set_memory_growth(physical_devices[0], True)
#endregion


''' WEEK 04 '''


""" HYPERPARAMETER TUNNING """
# >> See slide 


# In[1]: BATCH NORMALIZATION
#region
# 1.1. A NN with BN layers (added after the Activation function)
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.BatchNormalization(), # replace the feature scalers
    keras.layers.Dense(300, activation="relu", use_bias=False), # remove 300 bias, b/c BN already has
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="relu", use_bias=False), # remove 100 bias
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation="softmax")
])
model.summary()

# >> See slide for Non-trainable params

#%% 1.2. Another NN with BN layers (added before the Activation function)
# NOTE: BN added BEFORE activation usually gives BETTER result.
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300, use_bias=False), 
    keras.layers.BatchNormalization(), # add BN before Activation
    keras.layers.Activation("relu"),
    keras.layers.Dense(100, use_bias=False),  
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.Dense(10, activation="softmax")
])
model.summary()
#endregion
