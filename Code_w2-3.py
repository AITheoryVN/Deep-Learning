''' 
Main reference: github/ageron/handson-ml2/blob/master/10_neural_nets_with_keras.ipynb
Last review: Feb 2022
'''

# In[00]: IMPORTS AND COMMON SETTINGS
#region
from ast import dump
from os.path import join
import joblib
import shutil  
import sys
import time
from numpy.core.numeric import Inf
assert sys.version_info >= (3, 5) # Python ≥3.5 is required
import sklearn
assert sklearn.__version__ >= "0.20" # Scikit-Learn ≥0.20 is required
import tensorflow as tf
assert tf.__version__ >= "2.0" # TensorFlow ≥2.0 is required
from tensorflow import keras
import numpy as np
import os
np.random.seed(42) # to make THE output stable across runs
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd") # Ignore useless warnings (see SciPy issue #5998)
#endregion


''' WEEK 02 '''

# In[01]: LOAD and EXPLORE FASHION MNIST DATA
#region
# 1.1. Load data
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
X_train_full.shape #60,000 grayscale images, each 28x28 pixels
X_train_full.dtype #pixel intensity is represented as a byte (0 to 255)
y_train_full #labels are the class IDs: from 0 to 9
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]             
# 1.2. Scale the pixel intensities down to the 0-1 range:
X_train_full = X_train_full / 255
X_test = X_test / 255
# 1.3. Split a validation set from training set:
X_valid, X_train = X_train_full[:5000], X_train_full[5000:] 
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_valid.shape
X_train.shape
# 1.4. Plot a sample:
plt.imshow(X_train[0], cmap="binary")
plt.axis('off')
plt.show()
# 1.5. Plot several samples with labels:
if 1:
    n_rows = 4
    n_cols = 10
    plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
    for row in range(n_rows):
        for col in range(n_cols):
            index = n_cols * row + col
            plt.subplot(n_rows, n_cols, index + 1)
            plt.imshow(X_train[index], cmap="binary", interpolation="nearest")
            plt.axis('off')
            plt.title(class_names[y_train[index]], fontsize=12)
    #plt.subplots_adjust(wspace=0.2, hspace=0.5)
    plt.tight_layout(False)
    plt.savefig('figs/fashion_mnist_plot', dpi=300)
    plt.show()
#endregion


""" SEQUENTIAL API """

# In[02]: CREATE A MODEL
#region
# Info: "None" (in Output shape) means the batch size (#rows) can be anything
# Use 1 in 3 following Syntax
# Syntax 1:
if 1:
    model = keras.models.Sequential( # Sequential(): Feed-forward model
    [
        keras.layers.Flatten(input_shape=[28, 28]), # Input layer: flatten a 2D-array to 1D-array (784 neurons)
        keras.layers.Dense(300, activation="relu"), # Fully-connected (FC) layer, 300 neurons
        keras.layers.Dense(100, activation="relu"), # FC layer, 100 neurons
        keras.layers.Dense(10, activation="softmax"), # Output layer
    ])
    model.summary() 
# Syntax 2:
if 0:
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[28, 28]))
    model.add(keras.layers.Dense(300, activation="relu"))
    model.add(keras.layers.Dense(100, activation="relu"))
    model.add(keras.layers.Dense(10, activation="softmax"))
    model.summary()
# Syntax 3: (functional API (see later))
if 0:
    layer0 = keras.layers.Input(shape=[28, 28])
    layer1 = keras.layers.Flatten()
    layer2 = keras.layers.Dense(300, activation="relu")
    layer3 = keras.layers.Dense(100, activation="relu")
    layer4 = keras.layers.Dense(10, activation="softmax")
    model=keras.Model(inputs=layer0, outputs=layer4(layer3(layer2(layer1(layer0)))))
    model.summary()     

# >> QUESTION: Why are there 235500 parameters in the hidden layer 1?
# >> QUESTION: Is 235500 parameters too many? Overfitting?
#endregion


# In[03]: MANIPULATE THE MODEL
#region
# 3.1. Plot the model
keras.utils.plot_model(model, "figs/my_fashion_mnist_model.png", show_shapes=True)
# 3.2. Get parameters of a layer 
    # Get a layer by index
input_layer = model.layers[0]
input_layer.name
    # Get a layer by its name  
dense02 = model.get_layer('dense_6')  # name is shown by model.summary()
    # Get parameters of a layer
weights, biases = dense02.get_weights()
weights.shape
biases
# 3.3. Set parameters of a layer
weights = np.random.rand(*weights.shape) # *tuple: unpack the tuple 
biases = np.random.rand(*biases.shape)
biases[0]=11
biases[2]=22
dense02.set_weights([weights, biases])
#endregion


# In[04]: COMPILE AND TRAIN
#region
# 4.1. Compiling: specify loss, optimizer, metrics
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(lr=0.01), # Stochastic Gradient Descent
              metrics=["accuracy"])

# >> See slide loss functions info

# 4.2. Training:
n_epochs=30
new_training=0
if new_training:
    # Info: history object
    #   history.params
    #   history.epoch
    #   history.history: dictionary of loss and metrics
    history = model.fit(X_train, y_train, epochs=n_epochs,
                    #class_weight={0: 2, 1: 3.4, 2: 11, 3: 1, 4: 1, 5: 1, 6: 22, 7: 248, 8: 1, 9: 1}, # use this if training data is very imbalanced (to give a larger weight to underrepresented classes)
                    #sample_weight=1, # use this if some instances were labeled by experts while others by crowdsourcing platforms
                    #validation_split=0.1) # ask Keras to use 10% of training set for validation
                    validation_data=(X_valid, y_valid)) # provide a validatation set
    model.save('model01') # save as a folder
    #model.save('model01.h5') # save as a HDF5 format
    history = history.history # store only this
    joblib.dump(history,'model01/history01')
else:
    model = keras.models.load_model('model01')
    history = joblib.load('model01/history01')
history.keys()
history['accuracy']
history['val_accuracy'][n_epochs-1]
#endregion


# In[05]: LEARNING CURVES
#region
# 5.1. Plot learning curves
import pandas as pd
#pd.DataFrame(history).plot(figsize=(8, 5))
#pd.DataFrame({'Training loss':history['loss'],'Validation loss':history['val_loss']}).plot(figsize=(8, 5))
pd.DataFrame({'Training acc':history['accuracy'],'Validation acc':history['val_accuracy']}).plot(figsize=(8, 5))
plt.grid(True)
plt.xlim(0, n_epochs)
plt.ylim(0, 1)
plt.xlabel('epoch')
plt.xticks(np.arange(0,n_epochs+1,5))
plt.savefig("figs/learning_curves_01")
plt.show()

# >> QUESTION: Is this NN underfitting or overfitting?

# 5.2. More training
# NOTE: 
#   fit() always continues training with the current weights 
#   If you want to re-initialize weights, the best way now is to declare the model again before calling fit()
n_epochs=20
new_training=0
if new_training:
    history2 = model.fit(X_train, y_train, epochs=n_epochs,
                    validation_data=(X_valid, y_valid))  
    model.save('model02')
    history2 = history2.history # store only this
    joblib.dump(history2,'model02/history02')
else:
    model = keras.models.load_model('model02')
    history2 = joblib.load('model02/history02')
history2['val_accuracy'][n_epochs-1]
#endregion


# In[06]: MAKE PREDICTIONS
#region
X_new = X_test[0:15]
y_proba = model.predict(X_new).round(2) # return probabilities (output of output neurons)
print('Prediction proba: \n', y_proba)
y_pred = model.predict_classes(X_new) # return class with highest proba
print('Predicted class:', y_pred)
print('True labels:    ', y_test[0:15])
#endregion


# In[07]: EVALUATE MODELS
#region
model = keras.models.load_model('model01')
model.evaluate(X_test, y_test)
model = keras.models.load_model('model02')
model.evaluate(X_test, y_test)
#endregion


# In[08]: REGRESSION WITH NN
#region
# 8.1. Load housing data and split into training, test and validation sets
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)
X_train.shape

# 8.2. Preprocess data (just scaling)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid) # Note that transform() is used instead of fit_transform()
X_test = scaler.transform(X_test) 

#%% 8.3. Create and train a NN model
# NOTE: to do regession
#   Output layer: 1 neuron, no activation
#   Loss function: MSE
model = keras.models.Sequential(
[
    keras.layers.Input(shape=X_train.shape[1]),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(1)
])
model.summary()
keras.utils.plot_model(model, "figs/housing_model.png", show_shapes=True)
model.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(lr=1e-3))
new_training=0
if new_training:
    np.random.seed(42)
    tf.random.set_seed(42)
    history3 = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))
    model.save('model03')
    history3 = history3.history # store only this
    joblib.dump(history3,'model03/history03')
else:
    model = keras.models.load_model('model03')
    history3 = joblib.load('model03/history03')
print(history3["val_loss"][-1])

# 8.4. Plot learning curves, predict, evaluate
# Plot learning curves:
pd.DataFrame(history3).plot()
plt.show()
# Predict:
y_pred = model.predict(X_test[:10]).round(5)
print('prediction: ', y_pred.T)
print('True values: ', y_test[:10])
# Evaluate the model:
model_eval=model.evaluate(X_test, y_test)
print("Model MSE: ", model_eval)
#endregion


""" FUNCTIONAL API """

# In[09]: WIDE AND DEEP NN
#region
# 9.1. Create a NN
# Syntax 1:
if 1:
    input = keras.layers.Input(shape=X_train.shape[1:])
    hidden1 = keras.layers.Dense(30, activation="relu")(input)
    hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
    concat = keras.layers.concatenate([input, hidden2]) # (see fig) 30 col + 8 col = 38 columns
    output = keras.layers.Dense(1)(concat) # (see fig) 38 col => 1 col
    model = keras.models.Model(inputs=[input], outputs=[output])
    model.summary()
    keras.utils.plot_model(model, "figs/wide_deep.png", show_shapes=True)
# Syntax 2:
if 0:
    input = keras.layers.Input(shape=X_train.shape[1:])
    hidden1 = keras.layers.Dense(30, activation="relu")
    hidden2 = keras.layers.Dense(30, activation="relu")
    concat = keras.layers.concatenate([input, hidden2(hidden1(input))])
    output = keras.layers.Dense(1)
    model = keras.models.Model(inputs=[input], outputs=[output(concat)])
    model.summary()
    keras.utils.plot_model(model, "figs/wide_deep.png", show_shapes=True)

# 9.2. Complile and train (the same as section [08])
model.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(lr=1e-3))
new_training=0
if new_training:
    np.random.seed(42)
    tf.random.set_seed(42)
    history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))
    model.save('model04')
    history = history.history # store only this
    joblib.dump(history,'model04/history04')
else:
    model = keras.models.load_model('model04')
    history = joblib.load('model04/history04')
print(history["val_loss"][-1])

# 9.3. Plot learning curves, predict, evaluate
# Plot learning curves:
pd.DataFrame(history).plot()
plt.show()
# Predict:
y_pred = model.predict(X_test[:10]).round(5)
print('prediction: ', y_pred.T)
print('True values: ', y_test[:10])
# Evaluate the model:
model_eval=model.evaluate(X_test, y_test)
print("Model MSE: ", model_eval)
#endregion


# In[10]: WIDE AND DEEP NN WITH MULTIPLE INPUTS
#region
# 10.1. Create a NN
input_A = keras.layers.Input(shape=[5], name="wide_input")
input_B = keras.layers.Input(shape=[6], name="deep_input")
hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([input_A, hidden2])
output = keras.layers.Dense(1, name="output")(concat)
model = keras.models.Model(inputs=[input_A, input_B], outputs=[output])
model.summary()
keras.utils.plot_model(model, "figs/wide_deep_2.png", show_shapes=True)

# 10.2. Complile and train 
model.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(lr=1e-3))
new_training=0
if new_training:
    np.random.seed(42)
    tf.random.set_seed(42)

    X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]
    X_valid_A, X_valid_B = X_valid[:, :5], X_valid[:, 2:]
    history = model.fit((X_train_A, X_train_B), y_train, epochs=100, validation_data=((X_valid_A, X_valid_B), y_valid))
    
    model.save('model05')
    history = history.history # store only this
    joblib.dump(history,'model05/history')
else:
    model = keras.models.load_model('model05')
    history = joblib.load('model05/history')
print(history["val_loss"][-1])

# 10.3. Plot learning curves, predict, evaluate
# Plot learning curves:
pd.DataFrame(history).plot()
plt.show()
# Predict:
X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:] 
n_predict = 7
X_new_A, X_new_B = X_test_A[:n_predict], X_test_B[:n_predict]

y_pred = model.predict((X_new_A, X_new_B)).round(5)
print('Prediction: ', y_pred.T)
print('True values: ', y_test[:n_predict])
# Evaluate the model:
model_eval=model.evaluate((X_test_A, X_test_B), y_test)
print("Model MSE: ", model_eval)
#endregion


# In[11]: WIDE AND DEEP NN WITH MULTIPLE OUTPUTS
#region
# 11.1. Create the NN
input_A = keras.layers.Input(shape=[5], name="wide_input")
input_B = keras.layers.Input(shape=[6], name="deep_input")
hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([input_A, hidden2])
output_1 = keras.layers.Dense(1, name="main_output")(concat)
output_2 = keras.layers.Dense(1, name="aux_output")(hidden2)
model = keras.models.Model(inputs=[input_A, input_B], outputs=[output_1, output_2])
model.summary()
keras.utils.plot_model(model, "figs/wide_deep_3.png", show_shapes=True)

# 11.2. Complile and train 
    # NOTE: 
    #   + Each output require a loss, hence a list of losses
    #   + Total loss is the weighted sum of losses
model.compile(loss=["mse", "mse"], loss_weights=[0.9, 0.1], optimizer=keras.optimizers.SGD(lr=1e-3))

new_training=0
if new_training:
    history = model.fit((X_train_A, X_train_B), (y_train, y_train), epochs=100,
                    validation_data=((X_valid_A, X_valid_B), (y_valid, y_valid)))
    model.save('model06')
    history = history.history # store only this
    joblib.dump(history,'model06/history')
else:
    model = keras.models.load_model('model06')
    history = joblib.load('model06/history')
print(history["val_loss"][-1])

# 11.3. Plot learning curves, predict, evaluate
# Plot learning curves:
n_epoch_plot = 50
for key in history.keys():
    history[key] = history[key][:n_epoch_plot]
pd.DataFrame(history).plot()
plt.show()
# Predict:
y_pred, y_hidden  = model.predict((X_new_A, X_new_B))
print('Prediction:    ', y_pred.round(5).T)
#print('Hidden output: ', y_hidden.round(5).T)
print('True values:    ', y_test[:n_predict])
# Evaluate the model:
model_eval=model.evaluate((X_test_A, X_test_B), (y_test, y_test))
print("total_loss, main_loss, aux_loss: ", [round(a,3) for a in model_eval])
#endregion


""" SUBCLASSING API """

# In[12]: WIDE AND DEEP NN by SUBCLASSING API
#region
# 12.1. Create the NN
class WideAndDeepModel(keras.models.Model):
    def __init__(self, units=30, activation="relu", **kwargs): #kwargs: keyword arguments
        super().__init__(**kwargs)
        self.hidden1 = keras.layers.Dense(units, activation=activation)
        self.hidden2 = keras.layers.Dense(units, activation=activation)
        self.main_output = keras.layers.Dense(1)
        self.aux_output = keras.layers.Dense(1)
        
    def call(self, inputs): 
        # INFO: Computation of the NN. 
        #       You can use loop, branching,... like normal functions
        input_A, input_B = inputs
        hidden1 = self.hidden1(input_B)
        hidden2 = self.hidden2(hidden1)
        concat = keras.layers.concatenate([input_A, hidden2])
        main_output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return main_output, aux_output
model = WideAndDeepModel(units=30, activation="relu")

#%% 12.2. Complile and train 
model.compile(loss=["mse", "mse"], loss_weights=[0.9, 0.1], optimizer=keras.optimizers.SGD(lr=1e-3))
new_training=0
if new_training:
    history = model.fit((X_train_A, X_train_B), (y_train, y_train), epochs=20,
                    validation_data=((X_valid_A, X_valid_B), (y_valid, y_valid)))
    model.save('model07')
    history = history.history # store only this
    joblib.dump(history,'model07/history')
else:
    model = keras.models.load_model('model07')
    history = joblib.load('model07/history')
print(history["val_loss"][-1])

#%% 12.3. Plot learning curves, predict, evaluate
# Plot learning curves:
pd.DataFrame(history).plot()
plt.show()
# Predict:
y_pred, y_hidden = model.predict((X_new_A, X_new_B))
print('Prediction:    ', y_pred.round(5).T)
#print('Hidden output: ', y_hidden.round(5).T)
print('True values:    ', y_test[:n_predict])
# Evaluate the model:
model_eval=model.evaluate((X_test_A, X_test_B), (y_test, y_test))
print("total_loss, main_loss, aux_loss: ", [round(a,3) for a in model_eval])
#endregion


""" CALLBACKS """
# >> See slide

# In[13]: MODELCHECKPOINT and EARLYSTOPPING CALLBACKS 
# 13.1. Create a NN
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)
model = keras.models.Sequential([
    keras.layers.Input(shape = [8]),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(1) ])    
model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))

# 13.2. Use ModelCheckpoint callback to SAVE model while tranining
# Argument: 
#   save_best_only=True (require validation_data): only save if val_loss (or other metrics set by monitor=) is better
checkpoint_cb = keras.callbacks.ModelCheckpoint("model08/best.h5", monitor='val_loss', save_best_only=True) # "model08/epoch{epoch:02d}-val_loss{val_loss:.2f}.h5" 

new_training=0
if new_training!=0:
    history = model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid),
                        callbacks=[checkpoint_cb])
    model.save('model08/last.h5')
else:
    model = keras.models.load_model('model08/last.h5')
print('mse_val_last = %.2f' % model.evaluate(X_valid, y_valid))
model08_best = "model08/best.h5" # "model08/epoch86-val_loss0.34.h5"
model = keras.models.load_model(model08_best) # rollback to best model after training
print('mse_val_best = %.2f' % model.evaluate(X_valid, y_valid)) # model.evaluate(X_test, y_test)

#%% 13.3. Use EarlyStopping to STOP training when there is no improvement on val_loss for a number of epochs 
# Arguments: 
#   patience: a number of epochs to wait before stop training
#   restore_best_weights=True: auto rollback to the best model after training
# NOTE: good practice is to combine both callbacks
#   ModelCheckpoint: to save model frequently (in case the computer crashes)
#   EarlyStopping: to stop training early (to avoid wasting time and resources)
model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))
early_stopping_cb = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
checkpoint_cb = keras.callbacks.ModelCheckpoint("model08/best.h5", save_best_only=True) 
history = model.fit(X_train, y_train, epochs=100,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb, early_stopping_cb])

#%% 13.4. Custom callbacks
# INFO: 
#   Just write methods: on_train_begin(), on_train_end(), on_epoch_begin(), on_epoch_end()...
#   logs: a dict containing the metrics results.
#   See more at https://www.tensorflow.org/guide/keras/custom_callback 
class PrintBestEpoch(keras.callbacks.Callback):
    # This callback prints the best epoch when the tranining is done
    def __init__(self):
        super().__init__()
        self.best_val_loss=Inf 
        self.best_epoch=0
    def on_epoch_end(self, epoch, logs):
        if logs["val_loss"]<self.best_val_loss:
            self.best_val_loss=logs["val_loss"]
            self.best_epoch=epoch+1 # epoch id starts at 0
    def on_train_end(self, logs):
        print("\nBest epoch: %2d, best val_loss: %.4f" % (self.best_epoch, self.best_val_loss))
print_best_epoch_cb = PrintBestEpoch()
history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_valid, y_valid),
                    callbacks=[print_best_epoch_cb])        
#endregion


# In[14]: TENSORBOARD
#region
# NOTE: to use TensorBoard
#   1. Create a log directory, e.g., "LOGS"
#   2. Use TensorBoard callback to write logs of runs to this log's subfolders (each run in a different folder).
#   3. Start TensorBoard server by opening the Windows Explorer at the log folder and run this command (in the Address bar): tensorboard --logdir=LOGS --port=5006
#   4. Open web browser with the provided link, e.g., http://LAPTOP-QUANG:5006
# INFO: In TensorBoard terminologies:
#   A log file is call "event file".
#   A record (in an event file) is called a "summary".

# 14.1. Create a NN
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=[8]),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(1) ])    
model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))

# 14.2. Create logs folders name (different for each run)
root_logdir = os.path.join(os.curdir, "logs")
def run_logdir_name_by_time(prefix=''):
    run_id = time.strftime(prefix + "run_date%Y_%m_%d-time%H_%M_%S")
    return os.path.join(root_logdir, run_id)

# 14.3. Use TensorBoard callback to log the training
tensorboard_cb = keras.callbacks.TensorBoard(log_dir=run_logdir_name_by_time())
history = model.fit(X_train, y_train, epochs=3,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb, tensorboard_cb])


#%% 14.4. Use SummaryWriter create_file_writer()
# NOTE: every thing written to files by create_file_writer() can be displayed by TensorBoard server. https://www.tensorflow.org/api_docs/python/tf/summary/SummaryWriter
logs_others_name = run_logdir_name_by_time(prefix="logs_others_") 
writer = tf.summary.create_file_writer(logdir=logs_others_name)
with writer.as_default(): 
    for step in range(1, 3): 
        # Write some scalar to file:
        data = np.sin(step/10)
        tf.summary.scalar("my_scalar", data, step=step) 
        # Save some histogram:
        data = np.random.randn(100)*step # some random data         
        tf.summary.histogram("my_hist_%d" % step, data, step=step) 
        # Save some images:
        images = np.random.rand(5, 32, 32, 3)*step  # random 32×32 RGB images. A Tensor representing pixel data with shape [k, h, w, c], where k is the number of images, h and w are the height and width of the images, and c is the number of channels. https://www.tensorflow.org/api_docs/python/tf/summary/image
        tf.summary.image("my_images", images, step=step, max_outputs=10) 
        # Store some texts:
        texts = ["Step: " + str(step), "Text (step^2): " + str(step**2)] 
        tf.summary.text("my_text", texts, step=step) 
        # Keep some audio:
        sine_wave = tf.math.sin(tf.range(12000) / 48000 * 2 * np.pi * step) 
        audio = tf.reshape(tf.cast(sine_wave, tf.float32), [1, -1, 1]) 
        tf.summary.audio("my_audio", audio, sample_rate=48000, step=step)
writer.close()
#endregion




''' DONE '''

