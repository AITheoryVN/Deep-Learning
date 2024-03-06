''' 
Main reference: Chapter 15 in (Géron, 2019)
Last review: May 2022
'''

# In[0]: IMPORTS AND FUNCTIONS
#region
import sys
from tensorflow import keras
from tensorflow.core.protobuf.cluster_pb2 import JobDef
from tensorflow.keras import callbacks, layers
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.layers.core import Flatten
from tensorflow.python.keras.layers.pooling import AvgPool1D, AvgPool2D
from tensorflow.python.ops.gen_array_ops import shape
assert sys.version_info >= (3, 5) # Python ≥3.5 is required
import sklearn
assert sklearn.__version__ >= "0.20" # Scikit-Learn ≥0.20 is required
import tensorflow as tf
assert tf.__version__ >= "2.0" # TensorFlow ≥2.0 is required
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
np.random.seed(42)
tf.random.set_seed(42)
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
import joblib
    
use_GPU = 0
if use_GPU:
	physical_devices = tf.config.list_physical_devices('GPU')
	tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Don't worry if you don't understand below functions. Just go down the code and you will see.
def generate_time_series(no_of_series, no_of_time_steps):
	''' Generate univarate time series.
	Returns np array of shape: [no_of_series, no_of_time_steps, 1]
	'''
	freq1, freq2, offsets1, offsets2 = np.random.rand(4, no_of_series, 1)
	time = np.linspace(0, 1, no_of_time_steps)
	series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  #   wave 1
	series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # + wave 2
	series += 0.1 * (np.random.rand(no_of_series, no_of_time_steps) - 0.5)   # + noise
	return series[..., np.newaxis].astype(np.float32)

def plot_series(series, y=None, y_pred=None, n_steps=50, n_future_steps=1, x_label="$t$", y_label="$x(t)$", marker='o', color='blue'):
    legends = []
    if y is not None:
    	plt.plot(n_steps+n_future_steps-1, y, "go", markersize=8)
    	legends.append('Future value')
    if y_pred is not None:
    	plt.plot(n_steps+n_future_steps-1, y_pred, "rx", markersize=8, markeredgewidth=3)
    	legends.append('Predicted value')
    plt.grid(True)
    
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16, rotation=0)
    plt.plot(series, color=color, marker=marker, linestyle='-') # plot series
    plt.legend(legends)
    plt.hlines(0, 0, 100, linewidth=1)
    plt.axis([0, n_steps+n_future_steps+1, -1, 1])

def plot_multiple_forecasts(X, Y, Y_pred=None):
    n_steps = X.shape[0]
    n_future_steps = Y.shape[0]
    plot_series(X)
    plt.plot(np.arange(n_steps, n_steps + n_future_steps), Y, "go-", label="Actual")
    if Y_pred is not None:
        plt.plot(np.arange(n_steps, n_steps + n_future_steps), Y_pred, "rx-", markersize=8, markeredgewidth=2, label="Forecast")
    plt.axis([0, n_steps + n_future_steps, -1, 1])
    plt.legend(fontsize=14)

def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])

#endregion


''' WEEK 12 '''


# In[1]: RNNs FOR TIME SERIES: PREDICTING ONE NEXT VALUE
#region
# 1.1. Generate the dataset
# NOTE: 
# 	+ Dataset of multivariate time series has shape: 
# 	  [no_of_series, no_of_time_steps, no_of_values]
# 	+ Dataset of univariate time series has shape: 
# 	  [no_of_series, no_of_time_steps, 1]
def generate_time_series(no_of_series, no_of_time_steps):
	''' Generate univarate time series.
	Returns np array of shape: [no_of_series, no_of_time_steps, 1]
	'''
	freq1, freq2, offsets1, offsets2 = np.random.rand(4, no_of_series, 1)
	time = np.linspace(0, 1, no_of_time_steps)
	series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  #   wave 1
	series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # + wave 2
	series += 0.1 * (np.random.rand(no_of_series, no_of_time_steps) - 0.5)   # + noise
	return series[..., np.newaxis].astype(np.float32)

np.random.seed(42)
n_steps = 50
n_future_steps = 1
series = generate_time_series(10000, n_steps + n_future_steps)
X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, y_test = series[9000:, :n_steps], series[9000:, -1]

# 1.2. Plot time series
def plot_series(series, y=None, y_pred=None, n_steps=50, n_future_steps=1, x_label="$t$", y_label="$x(t)$", marker='o', color='blue'):
    legends = []
    if y is not None:
    	plt.plot(n_steps+n_future_steps-1, y, "go", markersize=8)
    	legends.append('Future value')
    if y_pred is not None:
    	plt.plot(n_steps+n_future_steps-1, y_pred, "rx", markersize=8, markeredgewidth=3)
    	legends.append('Predicted value')
    plt.grid(True)
    
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16, rotation=0)
    plt.plot(series, color=color, marker=marker, linestyle='-') # plot series
    plt.legend(legends)
    plt.hlines(0, 0, 100, linewidth=1)
    plt.axis([0, n_steps+n_future_steps+1, -1, 1])

n_series_to_plot = 3
for i in range(n_series_to_plot):
	plot_series(series=X_valid[i, :, 0], y=y_valid[i, 0], n_steps=50)
	plt.title('Series '+str(i))
	plt.show()


# >>> TASK TO DO: Predict next value y_t+1


#%% 1.2. Compute Baseline metrics
# Baseline model 1. Naive forcasting: prediction = last value of the series
y_pred = X_valid[:, -1]
mse_naive = np.mean(keras.losses.mean_squared_error(y_valid, y_pred))
print('\nMSE of naive forcasting:', np.round(mse_naive,4))
n_series_to_plot = 2
for i in range(n_series_to_plot):
	plot_series(series=X_valid[i, :, 0], y=y_valid[i, 0], y_pred=y_pred[i], n_steps=50)
	plt.title('Series '+str(i))
	plt.show() 
print('\nMSE of naive forcasting:', np.round(mse_naive,4))

#%% Baseline model 2. Linear regression by a fully-connected (feed-forward) NN with 1 neuron (treats 50 steps as 50 features)
np.random.seed(42)
tf.random.set_seed(42)
model = keras.models.Sequential([
    #keras.layers.Flatten(input_shape=[50, 1]),
    keras.layers.Input(shape=[50]),
    keras.layers.Dense(1,) ]) # no activation
model.summary()
model.compile(loss="mse", optimizer="nadam")
#model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))
history = model.fit(X_train[:,:,0], y_train[:,0], epochs=20, validation_data=(X_valid[:,:,0], y_valid[:,0]))
mse_nn = history.history['val_loss'][-1]
# Evaluate NN model:
y_pred = model.predict(X_valid[:, :, 0])
n_series_to_plot = 5
for i in range(n_series_to_plot):
	plot_series(series=X_valid[i, :, 0], y=y_valid[i, 0], y_pred=y_pred[i], n_steps=50)
	plt.title('Series '+str(i))
	plt.show() 
print('\nMSE of simple FC net:', np.round(mse_nn,4))

#%% 1.3. Use simple RNN: sequence-to-vector network (b/c we predict 1 output)
# INFO: 
# 	+ Input shape of RNN: [n_steps, n_values], but RNN can handle sequence of any length, hence 1st dim = None.
# 	+ SimpleRNN(): 1 neuron gets 2 inputs: x_t and h_t-1. NOTE: in SimpleRNN, cell state h_t-1 = y_t-1.
#	+ By default, keras RNN returns ONLY the last output. If you want outputs at every time step: set return_sequences=True. 
np.random.seed(42)
tf.random.set_seed(42)
model = keras.models.Sequential([
	keras.layers.Input(shape=[None, 1]), # RNN can handle sequence of any length, hence 1st dim = None.
    keras.layers.SimpleRNN(units=1, return_sequences=False) ])
model.summary() 

# >>> QUESTION: What are the 3 params of this model?

new_training = 0
if new_training:
	model.compile(loss="mse", optimizer="nadam")
	history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))
	model.save(r'models/simpleRNN.h5')	
model = keras.models.load_model(r'models/simpleRNN.h5')

# Evaluate the model:
mse_rnn = model.evaluate(X_valid, y_valid)
y_pred = model.predict(X_valid)
n_series_to_plot = 3
for i in range(n_series_to_plot):
	plot_series(series=X_valid[i, :, 0], y=y_valid[i, 0], y_pred=y_pred[i], n_steps=50)
	plt.title('Series '+str(i))
	plt.show() 
print('\nMSE of simple RNN (3 params):', np.round(mse_rnn,4))
print('\nFor comparison:')
print('MSE of naive forcasting (0 param):', np.round(mse_naive,4))
print('MSE of simple FC net (51 params):', np.round(mse_nn,4))


#%% 1.4. Deep RNN: sequence-to-vector network
# NOTE: 
# 	+ MUST set return_sequences=True to generate output in every time step (not just the last x_t)
# 	+ 1 input x_t => N outputs (if RNN layer has N neurons): RNN layers 'expand' the input.
np.random.seed(42)
tf.random.set_seed(42)
model = keras.models.Sequential([
	keras.layers.Input(shape=[None, 1]),
    keras.layers.SimpleRNN(2, return_sequences=True),
    keras.layers.SimpleRNN(2, return_sequences=True),
    keras.layers.SimpleRNN(2, return_sequences=True),
    keras.layers.SimpleRNN(2, return_sequences=False), # b/c we only want ONE output at the LAST input x_t => set return_sequences=False
    keras.layers.Dense(1) ])
model.summary()
new_training = 0
if new_training:
	model.compile(loss="mse", optimizer="nadam")
	# NOTE: deeper net may require longer training.
	history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))
	model.save(r'models/deepRNN01.h5')
model = keras.models.load_model(r'models/deepRNN01.h5')

# Evaluate the model:
mse_deepRNN = model.evaluate(X_valid, y_valid)
y_pred = model.predict(X_valid)
n_series_to_plot = 3
for i in range(n_series_to_plot):
	plot_series(series=X_valid[i, :, 0], y=y_valid[i, 0], y_pred=y_pred[i], n_steps=50)
	plt.title('Series '+str(i))
	plt.show() 
print('\nMSE of deep RNN (41 params):', np.round(mse_deepRNN,4))
print('\nFor comparison:')
print('MSE of simple RNN (3 params):', np.round(mse_rnn,4))
print('MSE of simple FC NN (51 params):', np.round(mse_nn,4))

#endregion


# In[2]: RNNs FOR TIME SERIES: PREDICTING VALUE AFTER N TIME STEPS

# >>> See slide

#region
# 2.1. Forecasting single value after N future steps
#	   => Solution: Just change the training label (from next value to value after N steps)
# Generate data
np.random.seed(42)
n_steps = 50
n_future_steps = 10
series = generate_time_series(10000, n_steps + n_future_steps)
X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, y_test = series[9000:, :n_steps], series[9000:, -1]
n_series_to_plot = 3
for i in range(n_series_to_plot):
	plot_series(series=X_valid[i, :, 0], y=y_valid[i, 0], n_steps=n_steps, n_future_steps=n_future_steps)
	plt.title('Series '+str(i))
	plt.show()

# Train RNN model (the same model as above)
np.random.seed(42)
tf.random.set_seed(42)
model = keras.models.Sequential([
	keras.layers.Input(shape=[None, 1]),
    keras.layers.SimpleRNN(2, return_sequences=True),
    keras.layers.SimpleRNN(2, return_sequences=True),
    keras.layers.SimpleRNN(2, return_sequences=True),
    keras.layers.SimpleRNN(2, return_sequences=False), # b/c we only want ONE output at the LAST input x_t => set return_sequences=False
    keras.layers.Dense(1) ])
model.summary()
new_training = 0
if new_training:
	model.compile(loss="mse", optimizer="nadam")
	history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))
	model.save(r'models/deepRNN02.h5')
model = keras.models.load_model(r'models/deepRNN02.h5')
mse_deepRNN2 = model.evaluate(X_valid, y_valid)

# Evaluate the model:
y_pred = model.predict(X_valid)
n_series_to_plot = 7
for i in range(n_series_to_plot):
	plot_series(series=X_valid[i, :, 0], y=y_valid[i, 0], y_pred=y_pred[i], n_steps=n_steps, n_future_steps=n_future_steps)
	plt.title('Series '+str(i))
	plt.show() 
print('\nMSE of RNN (predicting value after 10 future steps):', np.round(mse_deepRNN2,4))
print('\nFor comparison:')
print('MSE of RNN (predicting next value):', np.round(mse_deepRNN,4))


#%% 2.2. Forecasting next N values
# SOLUTION 1: Iterate a seq-to-vec network N times
# 	Repeat N times:
#		1. Predict 1 next value
# 		2. Add this value into input sequence, go to 1
# Generate data:
np.random.seed(42)
n_steps = 50
n_future_steps = 10
series = generate_time_series(10000, n_steps + n_future_steps)
X_train, Y_train = series[:7000, :n_steps], series[:7000, -n_future_steps:, 0]
X_valid, Y_valid = series[7000:9000, :n_steps], series[7000:9000, -n_future_steps:, 0]
X_test, Y_test = series[9000:, :n_steps], series[9000:, -n_future_steps:, 0]
def plot_multiple_forecasts(X, Y, Y_pred=None):
    n_steps = X.shape[0]
    n_future_steps = Y.shape[0]
    plot_series(X)
    plt.plot(np.arange(n_steps, n_steps + n_future_steps), Y, "go-", label="Actual")
    if Y_pred is not None:
        plt.plot(np.arange(n_steps, n_steps + n_future_steps), Y_pred, "rx-", markersize=8, markeredgewidth=2, label="Forecast")
    plt.axis([0, n_steps + n_future_steps, -1, 1])
    plt.legend(fontsize=14)
n_series_to_plot = 2
for i in range(n_series_to_plot):
	plot_multiple_forecasts(X=X_valid[i, :, 0], Y=Y_valid[i])
	plt.title('Series '+str(i))
	plt.show()

# Use trained RNN (model 1) and predict N values repeatedly:
model = keras.models.load_model(r'models/deepRNN01.h5')
X = X_valid
for future_step in range(n_future_steps):
	y_pred_1_step = model.predict(X)
	y_pred_1_step = y_pred_1_step[..., np.newaxis] # just to match dim of X: n_series, n_steps, n_values
	X = np.concatenate([X, y_pred_1_step], axis=1)
Y_pred = X[:, -n_future_steps:,0]
# Evaluate the model:
avg_mse_RNN = np.mean(keras.metrics.mean_squared_error(Y_valid, Y_pred))
n_series_to_plot = 3
for i in range(n_series_to_plot):
	plot_multiple_forecasts(X_valid[i,:,0], Y_valid[i], Y_pred[i])
	plt.title('Series '+str(i))
	plt.show()
print('\nAvg MSE of RNN (predicting next 10 values):', np.round(avg_mse_RNN,4))
print('\nFor comparison:')
Y_naive_pred = Y_valid[:, -1:]
# Y_naive_pred = np.array([[pred]*10 for pred in Y_valid[:, -1:]])[:,:,0]
avg_mse_naive=np.mean(keras.metrics.mean_squared_error(Y_valid, Y_naive_pred))
print('Avg MSE of naive forcasting (predicting next 10 values):', np.round(avg_mse_naive,4))


#%% SOLUTION 2: Seq-to-vec network that outputs N vectors
# 		Just replace Dense(1) by Dense(10)
#		(optional) Increase #RNN neurons since we do more predictions (10 instead of 1)
np.random.seed(42)
tf.random.set_seed(42)
model = keras.models.Sequential([
	keras.layers.Input(shape=[None, 1]),
    keras.layers.SimpleRNN(15, return_sequences=True),
    keras.layers.SimpleRNN(15, return_sequences=True),
    keras.layers.SimpleRNN(15, return_sequences=True),
    keras.layers.SimpleRNN(15, return_sequences=False), # b/c we only want ONE output at the LAST input x_t => set return_sequences=False
    keras.layers.Dense(10) ])
model.summary()
new_training = 0
if new_training:
	model.compile(loss="mse", optimizer="nadam")
	history = model.fit(X_train, Y_train, epochs=20, validation_data=(X_valid, Y_valid))
	model.save(r'models/deepRNN03.h5')
model = keras.models.load_model(r'models/deepRNN03.h5')
Y_pred = model.predict(X_valid)
# Evaluate the model:
avg_mse_RNN2 = np.mean(keras.metrics.mean_squared_error(Y_valid, Y_pred))
n_series_to_plot = 2
for i in range(n_series_to_plot):
	plot_multiple_forecasts(X_valid[i,:,0], Y_valid[i], Y_pred[i])
	plt.title('Series '+str(i))
	plt.show()
print('\nAvg MSE of RNN (Dense(10)):', np.round(avg_mse_RNN2,4))
print('\nFor comparison:')
print('Avg MSE of naive forcasting:', np.round(avg_mse_naive,4))
print('Avg MSE of RNN (Dense(1)):', np.round(avg_mse_RNN,4))


#%% SOLUTION 3: Seq-to-seq network that outputs N vectors
# 	At t=0: model outputs a 10-D vector (the forecasts for time steps 1 to 10)
#	At t=1: model outputs a 10-D vector (the forecasts for time steps 2 to 11)
#	At t=2: model outputs a 10-D vector (the forecasts for time steps 3 to 12)
#	...
#	=> Training label for each sample: a list of N 10-D vectors, where N is the length of the input sequence. 
# Prepare training labels:
np.random.seed(42)
n_steps = 50
n_future_steps = 10
Y = np.empty((10000, n_steps, n_future_steps))
for t in range(n_future_steps):
    Y[..., t] = series[..., t+1 : t+n_steps+1, 0]
Y_train = Y[:7000]
Y_valid = Y[7000:9000]
Y_test = Y[9000:]

# Create and train a RNN:
np.random.seed(42)
tf.random.set_seed(42)
model = keras.models.Sequential([
	keras.layers.Input(shape=[None, 1]),
    keras.layers.SimpleRNN(15, return_sequences=True),
    keras.layers.SimpleRNN(15, return_sequences=True),
    keras.layers.SimpleRNN(15, return_sequences=True),
    keras.layers.SimpleRNN(15, return_sequences=True), # NOTE: now we want output at every time step
    keras.layers.Dense(10) ])
model.summary()
def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])
model.compile(loss="mse", optimizer='nadam', metrics=[last_time_step_mse])
new_training = 0
if new_training:
	history = model.fit(X_train, Y_train, epochs=20,
						validation_data=(X_valid, Y_valid))
	#model.save(r'models/deepRNN04') # NOTE: using custom metrics causes error when loading model. Hence save Y_pred and X_valid, Y_valid.
	Y_pred = model.predict(X_valid)
	joblib.dump([Y_pred, Y_valid, X_valid],r'models/valid_data_RNN3')
#model = keras.models.load_model(r'models/deepRNN04.h5') # NOTE: using custom metrics causes error when loading model. 
[Y_pred, Y_valid, X_valid] = joblib.load(r'models/valid_data_RNN3')

# Evaluate the model:
avg_mse_RNN3 = np.mean(last_time_step_mse(Y_valid, Y_pred))
n_series_to_plot = 5
for i in range(n_series_to_plot):
	plot_multiple_forecasts(X_valid[i,:,0], Y_valid[i,-1], Y_pred[i,-1])
	plt.title('Series '+str(i))
	plt.show()
print('\nAvg MSE of RNN seq2seq (Dense(10)):', np.round(avg_mse_RNN3,4))
print('\nFor comparison:')
print('Avg MSE of RNN sed2vec (Dense(10)):', np.round(avg_mse_RNN2,4))
print('Avg MSE of RNN sed2vec (Dense(1)):', np.round(avg_mse_RNN,4))
print('Avg MSE of naive forcasting:', np.round(avg_mse_naive,4))

#endregion

# In[3]: [qADVANCE] RNNs with Layer Norm (to address unstable gradients problem in training with LONG sequences)
#region
# 3.1. Define a custom RNN neuron with Layer Norm 
from tensorflow.keras.layers import LayerNormalization
class LNSimpleRNNCell(keras.layers.Layer):
    def __init__(self, units, activation="tanh", **kwargs):
        super().__init__(**kwargs)
        self.state_size = units
        self.output_size = units
        self.simple_rnn_cell = keras.layers.SimpleRNNCell(units, activation=None)
        self.layer_norm = LayerNormalization()
        self.activation = keras.activations.get(activation)
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if inputs is not None:
            batch_size = tf.shape(inputs)[0]
            dtype = inputs.dtype
        return [tf.zeros([batch_size, self.state_size], dtype=dtype)]
    def call(self, inputs, states):
        outputs, new_states = self.simple_rnn_cell(inputs, states)
        norm_outputs = self.activation(self.layer_norm(outputs))
        return norm_outputs, [norm_outputs]

# 3.2. Use the custom neuron and create a model
np.random.seed(42)
tf.random.set_seed(42)
model = keras.models.Sequential([
	keras.layers.Input(shape=[None, 1]), 
    keras.layers.RNN(LNSimpleRNNCell(20), # NOTE: use RNN(cell) not SimpleRNN(units) layer
					 return_sequences=True), 
    keras.layers.RNN(LNSimpleRNNCell(20), return_sequences=True),
    keras.layers.Dense(10) ])
model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
history = model.fit(X_train, Y_train, epochs=1,
                    validation_data=(X_valid, Y_valid))


#%% 3.3. Creating a Custom RNN Class
class MyRNN(keras.layers.Layer):
    def __init__(self, cell, return_sequences=False, **kwargs):
        super().__init__(**kwargs)
        self.cell = cell
        self.return_sequences = return_sequences
        self.get_initial_state = getattr(
            self.cell, "get_initial_state", self.fallback_initial_state)
    def fallback_initial_state(self, inputs):
        return [tf.zeros([self.cell.state_size], dtype=inputs.dtype)]
    @tf.function
    def call(self, inputs):
        states = self.get_initial_state(inputs)
        n_steps = tf.shape(inputs)[1]
        if self.return_sequences:
            sequences = tf.TensorArray(inputs.dtype, size=n_steps)
        outputs = tf.zeros(shape=[n_steps, self.cell.output_size], dtype=inputs.dtype)
        for step in tf.range(n_steps):
            outputs, states = self.cell(inputs[:, step], states)
            if self.return_sequences:
                sequences = sequences.write(step, outputs)
        if self.return_sequences:
            return sequences.stack()
        else:
            return outputs

np.random.seed(42)
tf.random.set_seed(42)
model = keras.models.Sequential([
    MyRNN(LNSimpleRNNCell(20), return_sequences=True, input_shape=[None, 1]),
    MyRNN(LNSimpleRNNCell(20), return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(10)) ])
model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
history = model.fit(X_train, Y_train, epochs=1,
                    validation_data=(X_valid, Y_valid))

#endregion


''' WEEK 13 '''


# In[4]: Long-term memory RNN neurons
#region
# 4.0. Generate data
np.random.seed(42)
n_steps = 50
n_future_steps = 10
series = generate_time_series(10000, n_steps + n_future_steps)
X_train = series[:7000, :n_steps] 
X_valid = series[7000:9000, :n_steps] 
X_test = series[9000:, :n_steps] 
Y = np.empty((10000, n_steps, n_future_steps))
for t in range(n_future_steps):
    Y[..., t] = series[..., t+1 : t+n_steps+1, 0]
Y_train = Y[:7000]
Y_valid = Y[7000:9000]
Y_test = Y[9000:]
n_series_to_plot = 2
for i in range(n_series_to_plot):
	plot_multiple_forecasts(X=X_valid[i, :, 0], Y=Y_valid[i,-1,:])
	plt.title('Series '+str(i))
	plt.show()
 
#%% 4.3. RNNs with Conv1D layers
# NOTE: After Conv1D, input seq x's length REDUCES
#       ie, #time_steps decreases
#       => MUST reduce label seq y accordingly.
'''
Input:
              |-----2-----|     |-----5---...------|     |-----23----|
        |-----1-----|     |-----4-----|   ...      |-----22----|
  |-----0----|      |-----3-----|     |---...|-----21----|
X: 0  1  2  3  4  5  6  7  8  9  10 11 12 ... 42 43 44 45 46 47 48 49
Y: 1  2  3  4  5  6  7  8  9  10 11 12 13 ... 43 44 45 46 47 48 49 50
 =>10 11 12 13 14 15 16 17 18 19 20 21 22 ... 52 53 54 55 56 57 58 59

Output: of Conv1D with kernel size 4, stride 2, VALID padding
X:    0=>3  2=>5  4=>7                    ...               46=>49
Y:    4=>13 6=>15 8=>17                   ...               50=>59

Source: (Géron, 2019)
'''
# 4.3.1. SimpleRNNs with Conv1D layers
np.random.seed(42)
tf.random.set_seed(42)
filter_size = 4
strides = 2
model = keras.models.Sequential([
	keras.layers.Input(shape=[None, 1]),
    keras.layers.Conv1D(filters=10, kernel_size=filter_size, strides=strides, padding="valid"),     
    keras.layers.SimpleRNN(15, return_sequences=True),
    keras.layers.SimpleRNN(15, return_sequences=True),
    keras.layers.SimpleRNN(15, return_sequences=True),
    keras.layers.SimpleRNN(15, return_sequences=True), # NOTE: now we want output at every time step
    keras.layers.Dense(10) ])
model.summary()

# Reduce label seq:
Y_train_reduced = Y_train[:, filter_size-1::strides, :] # Y_train[:,3::2,:]
Y_valid_reduced = Y_valid[:, filter_size-1::strides, :] # Y_valid[:,3::2,:]

model.compile(loss="mse", optimizer='nadam', metrics=[last_time_step_mse])
new_training = 0
if new_training:
	early_stopper = keras.callbacks.EarlyStopping(monitor='last_time_step_mse',patience=4, restore_best_weights=True)
	history = model.fit(X_train, Y_train_reduced, epochs=30, validation_data=(X_valid, Y_valid_reduced), callbacks=[early_stopper])
	Y_pred = model.predict(X_valid)
	joblib.dump([Y_pred, Y_valid, X_valid],r'models/valid_data_SimpleRNN_Conv1D.joblib')
[Y_pred, Y_valid, X_valid] = joblib.load(r'models/valid_data_SimpleRNN_Conv1D.joblib')
# Evaluate the model:
avg_mse_SimpleRNN_Conv1D = np.mean(last_time_step_mse(Y_valid_reduced, Y_pred[:, filter_size-1::strides, :]))
n_series_to_plot = 1
for i in range(n_series_to_plot):
	plot_multiple_forecasts(X_valid[i,:,0], Y_valid[i,-1], Y_pred[i,-1])
	plt.title('Series '+str(i))
	plt.show()
print('\nAvg MSE of SimpleRNN with Conv1D:', np.round(avg_mse_SimpleRNN_Conv1D,4))
print('\nFor comparison:')
[Y_pred_old, Y_valid_old, X_valid_old] = joblib.load(r'models/valid_data_RNN3')
avg_mse_RNN3 = np.mean(last_time_step_mse(Y_valid_old, Y_pred_old))
print('Avg MSE of SimpleRNN (no Conv1D)):', np.round(avg_mse_RNN3,4))


#%% 4.3.2. LSTMs with Conv1D layers
np.random.seed(42)
tf.random.set_seed(42)
model = keras.models.Sequential([
	keras.layers.Input(shape=[None, 1]),
    keras.layers.Conv1D(filters=10, kernel_size=filter_size, strides=strides, padding="valid"),     
    keras.layers.LSTM(15, return_sequences=True),
    keras.layers.LSTM(15, return_sequences=True),
    keras.layers.LSTM(15, return_sequences=True),
    keras.layers.LSTM(15, return_sequences=True), # NOTE: now we want output at every time step
    keras.layers.Dense(10) ])
model.summary()
model.compile(loss="mse", optimizer='nadam', metrics=[last_time_step_mse])
new_training = 0
if new_training:
	early_stopper = keras.callbacks.EarlyStopping(monitor='last_time_step_mse',patience=4, restore_best_weights=True)
	history = model.fit(X_train, Y_train_reduced, epochs=30, validation_data=(X_valid, Y_valid_reduced), callbacks=[early_stopper])
	Y_pred = model.predict(X_valid)
	joblib.dump([Y_pred, Y_valid, X_valid],r'models/valid_data_LSTM_Conv1D.joblib')
[Y_pred, Y_valid, X_valid] = joblib.load(r'models/valid_data_LSTM_Conv1D.joblib')
# Evaluate the model:
avg_mse_LSTM_Conv1D = np.mean(last_time_step_mse(Y_valid_reduced, Y_pred[:, filter_size-1::strides, :]))
n_series_to_plot = 1
for i in range(n_series_to_plot):
	plot_multiple_forecasts(X_valid[i,:,0], Y_valid[i,-1], Y_pred[i,-1])
	plt.title('Series '+str(i))
	plt.show()
print('\nAvg MSE of LSTM with Conv1D:', np.round(avg_mse_LSTM_Conv1D,4))
print('\nFor comparison:')
print('Avg MSE of SimpleRNN with Conv1D:', np.round(avg_mse_SimpleRNN_Conv1D,4))
print('Avg MSE of SimpleRNN (no Conv1D)):', np.round(avg_mse_RNN3,4))

#%% 4.3.3. GRUs with Conv1D layers
np.random.seed(42)
tf.random.set_seed(42)
model = keras.models.Sequential([
	keras.layers.Input(shape=[None, 1]),
    keras.layers.Conv1D(filters=10, kernel_size=filter_size, strides=strides, padding="valid"),     
    keras.layers.GRU(15, return_sequences=True),
    keras.layers.GRU(15, return_sequences=True),
    keras.layers.GRU(15, return_sequences=True),
    keras.layers.GRU(15, return_sequences=True), # NOTE: now we want output at every time step
    keras.layers.Dense(10) ])
model.summary()
model.compile(loss="mse", optimizer='nadam', metrics=[last_time_step_mse])
new_training = 0
if new_training:
	early_stopper = keras.callbacks.EarlyStopping(monitor='last_time_step_mse',patience=4, restore_best_weights=True)
	history = model.fit(X_train, Y_train_reduced, epochs=30, validation_data=(X_valid, Y_valid_reduced), callbacks=[early_stopper])
	Y_pred = model.predict(X_valid)
	joblib.dump([Y_pred, Y_valid, X_valid],r'models/valid_data_GRU_Conv1D.joblib')
[Y_pred, Y_valid, X_valid] = joblib.load(r'models/valid_data_GRU_Conv1D.joblib')
# Evaluate the model:
avg_mse_GRU_Conv1D = np.mean(last_time_step_mse(Y_valid_reduced, Y_pred[:, filter_size-1::strides, :]))
n_series_to_plot = 4
for i in range(n_series_to_plot):
	plot_multiple_forecasts(X_valid[i,:,0], Y_valid[i,-1], Y_pred[i,-1])
	plt.title('Series '+str(i))
	plt.show()
print('\nAvg MSE of GRU with Conv1D:', np.round(avg_mse_GRU_Conv1D,4))
print('\nFor comparison:')
print('Avg MSE of LSTM with Conv1D:', np.round(avg_mse_LSTM_Conv1D,4))
print('Avg MSE of SimpleRNN with Conv1D:', np.round(avg_mse_SimpleRNN_Conv1D,4))
print('Avg MSE of SimpleRNN (no Conv1D)):', np.round(avg_mse_RNN3,4))


#%% 4.4. (Simplified) WaveNet
# INFO: Dilation rate is the rate to expand the filter. Note that by expanding we mean to 'add holes' into the filter, NOT to increase the filter size (hence NOT to increase the #params). See examples at https://towardsdatascience.com/review-dilated-convolution-semantic-segmentation-9d5a5bd768f5
# NOTE:
#   + dilation_rate: ONLY works with 'stries=1' (for current tf version)
#   + padding="causal": left zero-padding to keep 
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential()
model.add(keras.layers.Input(shape=[None, 1]))
for rate in (1, 2, 4, 8, 16)*2:
    model.add(keras.layers.Conv1D(filters=40, kernel_size=2, padding="causal", activation="relu", dilation_rate=rate))
model.add(keras.layers.Conv1D(filters=10, kernel_size=1)) # output like Dense(10)
model.summary()
new_training = 0
if new_training:
    model.compile(loss="mse", optimizer="nadam", metrics=[last_time_step_mse])
    early_stopper = keras.callbacks.EarlyStopping(monitor='last_time_step_mse',patience=4, restore_best_weights=True)
    history = model.fit(X_train, Y_train, epochs=30, validation_data=(X_valid, Y_valid), callbacks=[early_stopper])
    Y_pred = model.predict(X_valid)
    joblib.dump([Y_pred, Y_valid, X_valid],r'models/valid_data_SimpleWaveNet.joblib')
[Y_pred, Y_valid, X_valid] = joblib.load(r'models/valid_data_SimpleWaveNet.joblib')
# Evaluate the model:
avg_mse_SimpleWaveNet = np.mean(last_time_step_mse(Y_valid_reduced, Y_pred[:, filter_size-1::strides, :]))
n_series_to_plot = 4
for i in range(n_series_to_plot):
	plot_multiple_forecasts(X_valid[i,:,0], Y_valid[i,-1], Y_pred[i,-1])
	plt.title('Series '+str(i))
	plt.show()
print('\nAvg MSE of simplified WaveNet:', np.round(avg_mse_SimpleWaveNet,4))
print('\nFor comparison:')
print('Avg MSE of GRU with Conv1D:', np.round(avg_mse_GRU_Conv1D,4))
print('Avg MSE of LSTM with Conv1D:', np.round(avg_mse_LSTM_Conv1D,4))
print('Avg MSE of SimpleRNN with Conv1D:', np.round(avg_mse_SimpleRNN_Conv1D,4))
print('Avg MSE of SimpleRNN (no Conv1D)):', np.round(avg_mse_RNN3,4))


#%% 4.5. (qAdvance) WaveNet
# The original WaveNet uses Gated Activation Units instead of ReLU and parametrized skip connections.
# Source: (Geron, 2019)
class GatedActivationUnit(keras.layers.Layer):
    def __init__(self, activation="tanh", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
    def call(self, inputs):
        n_filters = inputs.shape[-1] // 2
        linear_output = self.activation(inputs[..., :n_filters])
        gate = keras.activations.sigmoid(inputs[..., n_filters:])
        return self.activation(linear_output) * gate

def wavenet_residual_block(inputs, n_filters, dilation_rate):
    z = keras.layers.Conv1D(2 * n_filters, kernel_size=2, padding="causal", dilation_rate=dilation_rate)(inputs)
    z = GatedActivationUnit()(z)
    z = keras.layers.Conv1D(n_filters, kernel_size=1)(z)
    return keras.layers.Add()([z, inputs]), z

# WaveNet model:
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

n_layers_per_block = 3 # 10 in the paper
n_blocks = 1 # 3 in the paper
n_filters = 32 # 128 in the paper
n_outputs = 10 # 256 in the paper

inputs = keras.layers.Input(shape=[None, 1])
z = keras.layers.Conv1D(n_filters, kernel_size=2, padding="causal")(inputs)
skip_to_last = []
for dilation_rate in [2**i for i in range(n_layers_per_block)]*n_blocks:
    z, skip = wavenet_residual_block(z, n_filters, dilation_rate)
    skip_to_last.append(skip)
z = keras.activations.relu(keras.layers.Add()(skip_to_last))
z = keras.layers.Conv1D(n_filters, kernel_size=1, activation="relu")(z)
Y_proba = keras.layers.Conv1D(n_outputs, kernel_size=1, activation="softmax")(z)
model = keras.models.Model(inputs=[inputs], outputs=[Y_proba])
model.summary()

#%% Train WaveNet
model.compile(loss="mse", optimizer="nadam", metrics=[last_time_step_mse])
history = model.fit(X_train, Y_train, epochs=20, validation_data=(X_valid, Y_valid))


#endregion

 

