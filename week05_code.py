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


''' WEEK 05 '''

# In[2]: TRANSFER LEARNING
#region
# 2.1. Load fashion MNIST dataset
# Recall: Fashion MNIST consists of 70.000 images of 28x28 pixels each, 10 classes, e.g., “T-shirt”, “Trouser”, “Pullover”.
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# 2.2. Split the fashion MNIST training set into:
#   X_train_A: all images except for sandals and shirts (classes 5 and 6).
#   X_train_B: 50 images of sandals or shirts. 
def split_dataset(X, y):
    y_5_or_6 = (y == 5) | (y == 6) # sandals or shirts
    y_A = y[~y_5_or_6]
    y_A[y_A > 6] -= 2 # class indices 7, 8, 9 are moved to 5, 6, 7
    y_B = (y[y_5_or_6] == 6).astype(np.float32) # binary classification (class 5 = 0, class 6 = 1)
    return ((X[~y_5_or_6], y_A),
            (X[y_5_or_6], y_B))
(X_train_A, y_train_A), (X_train_B, y_train_B) = split_dataset(X_train, y_train)
(X_valid_A, y_valid_A), (X_valid_B, y_valid_B) = split_dataset(X_valid, y_valid)
(X_test_A, y_test_A), (X_test_B, y_test_B) = split_dataset(X_test, y_test)
X_train_B = X_train_B[:20]
y_train_B = y_train_B[:20]

#%% 2.3. Train (or load) model A
new_training = 0
if new_training:
    tf.random.set_seed(42)
    np.random.seed(42)
    
    model_A = keras.models.Sequential()
    model_A.add(keras.layers.Flatten(input_shape=[28, 28]))
    for n_hidden in (300, 100, 50, 50, 50):
        model_A.add(keras.layers.Dense(n_hidden, activation="elu", kernel_initializer='he_normal'))
    model_A.add(keras.layers.Dense(8, activation="softmax"))
    model_A.summary()
    tf.keras.utils.plot_model(model_A,'models/model_A.png',show_shapes=True)
    
    model_A.compile(loss="sparse_categorical_crossentropy",
                    optimizer=keras.optimizers.SGD(lr=1e-3),
                    metrics=["accuracy"])
    history = model_A.fit(X_train_A, y_train_A, epochs=20,
                        validation_data=(X_valid_A, y_valid_A))
    history = history.history
    joblib.dump(history,"models/model_A_history")
    model_A.save("models/model_A.h5")
else: 
    model_A = keras.models.load_model("models/model_A.h5")
    history = joblib.load("models/model_A_history")
history["val_accuracy"][-1]

#%% 2.4. Train model B FROM SCRATCH (with the same architecture as model A)
tf.random.set_seed(42)
np.random.seed(42)                
model_B_FROM_SCRATCH = keras.models.Sequential()
model_B_FROM_SCRATCH.add(keras.layers.Flatten(input_shape=[28, 28]))
for n_hidden in (300, 100, 50, 50, 50):
    model_B_FROM_SCRATCH.add(keras.layers.Dense(n_hidden, activation="elu", kernel_initializer='he_normal'))
model_B_FROM_SCRATCH.add(keras.layers.Dense(1, activation="sigmoid"))
model_B_FROM_SCRATCH.compile(loss="binary_crossentropy",
                optimizer=keras.optimizers.SGD(lr=1e-3),
                metrics=["accuracy"])
history = model_B_FROM_SCRATCH.fit(X_train_B, y_train_B, epochs=20,
                      validation_data=(X_valid_B, y_valid_B))
history = history.history
history["val_accuracy"][-1]

#%% 2.5. Train model B BY REUSING some hidden layers of model A
model_A = keras.models.load_model("models/model_A.h5")
tf.random.set_seed(42)
np.random.seed(42)
model_B_TRANSFER = keras.models.Sequential(model_A.layers[:-1]) # reuse layers of model A, except the last one (output layer, b/c #classes = 2)
# NOTE: the above declaration of model B shares model A's layers 
#       BY REFERENCE, hence model A's params will 
#       BE CHANGED when model B is trained.
#   To "deep copy" (i.e., clone) model A, use:
#       model_A_clone = keras.models.clone_model(model_A) # NOTE: this ONLY clones the ARCHITECTURE, NOT the params values.
#       model_A_clone.set_weights(model_A.get_weights()) # this clones the params values
model_B_TRANSFER.add(keras.layers.Dense(1, activation="sigmoid")) # add new output layer

for layer in model_B_TRANSFER.layers[:1]: # reuse parameters of the first 2 layers (ie. freeze training)
    layer.trainable = False

# NOTE: we MUST compile the model after freezing or unfreezing layers.   
model_B_TRANSFER.compile(loss="binary_crossentropy",
                     optimizer=keras.optimizers.SGD(lr=1e-3),
                     metrics=["accuracy"]) # may try reducing lr to reserve a bit more params of unfreezed layers
history = model_B_TRANSFER.fit(X_train_B,y_train_B, epochs=20,
                           validation_data=(X_valid_B, y_valid_B))
history = history.history
history["val_accuracy"][-1]

# NOTE: Transfer learning does NOT MUCH improve performance in small NNs. But for deep NNs (especially the CNNs) it is really helpful (see Chap 14 in (Géron, 2019)).

#endregion

 