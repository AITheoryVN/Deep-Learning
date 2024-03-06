''' 
Main reference: Chapter 14 in (Géron, 2019)
Last review: April 2022
'''

# In[0]: IMPORTS AND SETTINGS
#region
import sys
from tensorflow import keras
from tensorflow.core.protobuf.cluster_pb2 import JobDef
from tensorflow.keras import callbacks
from tensorflow.python.keras.layers.core import Flatten
from tensorflow.python.keras.layers.pooling import AvgPool1D, AvgPool2D
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

#endregion


''' WEEK 8 '''

# In[1]: CONVOLUTION OPERATIONS
#region
# 1.1. Load and preprocess images
img1_BGR = cv2.imread('images/HCMUTE.jpg')
img2_BGR = cv2.imread('images/PallasCat_zacalcatcollars.jpg')
# OpenCV stores images in BGR order instead of RGB, hence we reorder:
img1 = cv2.cvtColor(img1_BGR, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2_BGR, cv2.COLOR_BGR2RGB)
plt.imshow(img1) 
# Resize images:
img1 = cv2.resize(img1, (600,400))
img2 = cv2.resize(img2, (600,400))
# Scale pixel values to [0,1]: (b/c some layer ONLY allow float data)
img1 = img1/255
img2 = img2/255
# Batch images:
images = np.stack([img1, img2])
batch_size, height, width, channels = images.shape

# 1.2. Create 2 filters
filter_horizontal = np.zeros(shape=(7, 7, channels), dtype=np.float32)
filter_horizontal[3, :, :] = 1  # 3rd line pixels = 1
#filter_horizontal[:, :, 0].astype(np.int32) # just to show 
#plt.imshow(filter_horizontal[:, :, 0], cmap='gray')
filter_vertical = np.zeros(shape=(7, 7, channels), dtype=np.float32)
filter_vertical[:, 3, :] = 1  # 3rd column pixels = 1
filters = np.stack([filter_horizontal,filter_vertical], axis=3)
# NOTE: 
#   filters dim MUST be (width, height, channels, #filters)
#   images dim MUST be (#images, width, height, channels)
#   outputs dim (in code below): (#images, width, height, #feat maps)

# 1.3. Compute feature maps
# NOTE: padding must be "VALID" or "SAME":
#   + padding="VALID": NOT use zero padding. May ignore some rows and columns at the bottom and right of the input => receptive rectangle lies strictly within valid positions inside the input, hence the name "VALID".
#   + padding="SAME": uses zero padding if necessary. Output size = round-up(input size / stride), e.g., if input size = 13, stride = 5 => output size = 13/5 = 3 (round up from 2.6). If stride = 1 => output size = input size, hence the name "SAME"
outputs = tf.nn.conv2d(images, filters, strides=1, padding="SAME")
image_id = 1
feat_map_id = 0 # feat map 0: Horizontal filter
plt.imshow(outputs[image_id, :, :, feat_map_id], cmap="gray") 
plt.title('Horizontal filtering')
plt.axis('off')
plt.savefig('images/horizontial_filtered',dpi=300)
plt.show()
plt.imshow(outputs[image_id, :, :, 1], cmap="gray") # feat map 1: Vertical filter
plt.title('Vertical filtering')
plt.axis('off')
plt.savefig('images/vertical_filtered',dpi=300)
plt.show()
#endregion


# In[2]: POOLING OPERATION
#region
# 2.0. Load images
img1_BGR = cv2.imread('images/CanhDong_Hue_ivivu.jpg') # HCMUTE.jpg
img2_BGR = cv2.imread('images/PallasCat_zacalcatcollars.jpg')
img1 = cv2.cvtColor(img1_BGR, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2_BGR, cv2.COLOR_BGR2RGB)
# Resize images:
img1 = cv2.resize(img1, (600,400))
img2 = cv2.resize(img2, (600,400))
# Scale pixel values to [0,1]:
img1 = img1/255
img2 = img2/255
plt.imshow(img1) 
plt.title('Input')
plt.show()
images = np.stack([img1, img2])

# 2.1. Max poolings
max_pool = keras.layers.MaxPool2D(pool_size=5, strides=2) #pool_size=5 : 5x5 kernel, default stride=pool_size, default padding='VALID'
outputs = max_pool(images)
plt.imshow(outputs[0])
plt.title('Output of max pooling: pool_size=5, stride=2')
plt.show()
max_pool = keras.layers.MaxPool2D(pool_size=20, strides=2) 
outputs = max_pool(images)
plt.imshow(outputs[0])
plt.title('Output of max pooling: pool_size=20, stride=2')
plt.show()

#%% 2.2. Average poolings
avg_pool = keras.layers.AvgPool2D(pool_size=5, strides=2) 
outputs = avg_pool(images)
plt.imshow(img1) 
plt.title('Input')
plt.show()
plt.imshow(outputs[0])
plt.title('Output of avg pooling: pool_size=5, stride=2')
plt.show()
avg_pool = keras.layers.AvgPool2D(pool_size=20, strides=2) 
outputs = avg_pool(images)
plt.imshow(outputs[0])
plt.title('Output of avg pooling: pool_size=20, stride=2')
plt.show()

# [SKIP] qAdvance
# Pooling along the depth:
# depth_pool = keras.layers.Lambda(lambda X: tf.nn.max_pool(X, ksize=(1, 1, 1, 3), strides=(1, 1, 1, 3), padding="valid")) # ksize, strides: in (batch, height, width, depth). NOTE: depth MUST be a divisor of the input depth (eg, depth=3 WON'T work with input depth=20)
# with tf.device("/cpu:0"): # there is no GPU-kernel yet
#     depth_output = depth_pool(cropped_images)
# Global Average Pooling:
# global_avg_pool = keras.layers.GlobalAvgPool2D()
# global_avg_pool(images)

#endregion


# In[3]: CNNs vs. FULLY CONNECTED NETs  
#region
# 3.1. Load and preprocess Fashion MNIST
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)
plt.imshow(X_train[0], cmap='gray')
# Scale images: 
X_mean = X_train.mean(axis=0, keepdims=True)
X_std = X_train.std(axis=0, keepdims=True) + 1e-7
X_train = (X_train - X_mean) / X_std
X_valid = (X_valid - X_mean) / X_std
X_test = (X_test - X_mean) / X_std
# Add channel axis
X_train = X_train[:, :, :, np.newaxis]
X_valid = X_valid[:, :, :, np.newaxis]
X_test = X_test[:, :, :, np.newaxis]

#%% 3.2. Try a Fully connected net
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Flatten
from functools import partial
myDense = partial(Dense, activation='relu',kernel_initializer='glorot_normal')
FC_model = keras.Sequential([
    Input(shape=X_train.shape[1:]),
    Flatten(),
    myDense(units=400),
    BatchNormalization(),
    myDense(units=200),
    BatchNormalization(),
    myDense(units=100),
    BatchNormalization(),
    Dropout(0.5),
    myDense(units=10, activation='softmax') ])
FC_model.summary()
new_training = 0
if new_training:
    FC_model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
    model_saver = keras.callbacks.ModelCheckpoint('models/best_FC_model.h5',monitor='val_accuracy', save_best_only=True)
    early_stopper = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
    performance_sched = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.2)
    FC_model.fit(X_train, y_train, epochs=10, batch_size=32,
        validation_data=(X_valid, y_valid),
        callbacks=[model_saver, early_stopper, performance_sched])
FC_model = keras.models.load_model('models/best_FC_model.h5')
FC_model.evaluate(X_test, y_test)

#%% 3.3. Try a Convolutional net
from tensorflow.keras.layers import Conv2D, MaxPool2D
myConv2D = partial(Conv2D, kernel_size=(3,3), strides=(2,2), padding='SAME', activation='relu') # strides=1 b/c of small images
CNN_model = keras.Sequential([
    Input(shape=X_train.shape[1:]),
    myConv2D(filters=64, kernel_size=(5,5), strides=(2,2)),
    MaxPool2D(pool_size=(2,2)),
    #BatchNormalization(),
    myConv2D(filters=128),
    myConv2D(filters=128),    
    MaxPool2D(pool_size=(2,2)), 
    #BatchNormalization(),
    # myConv2D(filters=256),
    # myConv2D(filters=256),    
    # MaxPool2D(pool_size=(2,2)), 
    #BatchNormalization(),
    Flatten(),    
    myDense(units=100),    
    Dropout(0.5),
    myDense(units=10, activation='softmax') ])
CNN_model.summary()
new_training = 0
if new_training:
    CNN_model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
    model_saver = keras.callbacks.ModelCheckpoint('models/best_CNN_model_noBN.h5',monitor='val_accuracy', save_best_only=True)
    early_stopper = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
    performance_sched = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.2)
    #with tf.device('/gpu:0'):
    CNN_model.fit(X_train, y_train, epochs=5, batch_size=32,
        validation_data=(X_valid, y_valid),
        callbacks=[model_saver, early_stopper, performance_sched])
CNN_model = keras.models.load_model('models/best_CNN_model_noBN.h5')
CNN_model.evaluate(X_test, y_test)

# >>> RESULT:
#   FC model: 418,110 params, test acc: 88.3%
#   CNN model: 237,014 params, test acc: 91.3% 

#endregion

