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

 

''' WEEK 11 '''

# In[5]: SOTA CNN ARCHITECTURES (cont.)
#region
# 5.1. ResNet-34 (34-layer version)
from functools import partial
DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, strides=1, padding="SAME", use_bias=False)
class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            DefaultConv2D(filters, strides=strides),
            keras.layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(filters),
            keras.layers.BatchNormalization()]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConv2D(filters, kernel_size=1, strides=strides),
                keras.layers.BatchNormalization()]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)
model = keras.models.Sequential(name='my ResNet-34')
model.add(DefaultConv2D(64, kernel_size=7, strides=2, input_shape=[224, 224, 3]))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="SAME"))
prev_filters = 64
for filters in [64]*3 + [128]*4 + [256]*6 + [512]*3:
    strides = 1 if filters == prev_filters else 2
    model.add(ResidualUnit(filters, strides=strides))
    prev_filters = filters
model.add(keras.layers.GlobalAvgPool2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1000, activation="softmax"))
model.summary()

#%% 5.2. A try on Xception: use SeparableConv on ResNet-34
# NOTE: just replace Conv layer by SeparableConv: tf.keras.layers.SeparableConv2D() 
DefaultConv2D = partial(keras.layers.SeparableConv2D, kernel_size=3, strides=1, padding="SAME", use_bias=False)
class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            DefaultConv2D(filters, strides=strides),
            keras.layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(filters),
            keras.layers.BatchNormalization()]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConv2D(filters, kernel_size=1, strides=strides),
                keras.layers.BatchNormalization()]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)
model = keras.models.Sequential(name='my try on Xception with ResNet-34 architecture')
model.add(DefaultConv2D(64, kernel_size=7, strides=2, input_shape=[224, 224, 3]))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="SAME"))
prev_filters = 64
for filters in [64]*3 + [128]*4 + [256]*6 + [512]*3:
    strides = 1 if filters == prev_filters else 2
    model.add(ResidualUnit(filters, strides=strides))
    prev_filters = filters
model.add(keras.layers.GlobalAvgPool2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1000, activation="softmax"))
model.summary()

# >>NOTE: Look at the #params when using SeparableConv vs. Conv.: ~3 mil vs. ~21 mil!
#endregion 


# In[6]:  USE PRETRAINED MODELS from tf.keras
#region 
# 6.1. Load the model ResNet50
model = tf.keras.applications.resnet50.ResNet50(weights="imagenet")

#%% 6.2. Load and resize images
# NOTE: ResNet-50 expects 224x224-pixel images
import glob
file_names = glob.glob("images/*.jpg") # ['images/CanhDong_Hue_ivivu.jpg', 'images/PallasCat_zacalcatcollars.jpg', 'images/HCMUTE.jpg']
images = np.empty((0,224,224,3))
for file_name in file_names:
    img_BGR = cv2.imread(file_name)
    img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    #img = tf.image.resize(img, (224,224), antialias=True) 
    img = tf.image.resize_with_pad(img, 224,224, antialias=True) # NOTE: To resize but KEEP ASPECT RATIO: resize_with_pad() or crop_and_resize() 
    images = np.append(images, img[np.newaxis,:,:,:], axis=0)    
    if 10:
        plt.imshow(img/255) 
        plt.title(file_name)
        plt.show()
print(images.shape)
 
#%% 6.3. Preprocess the images 
# INFO: Images MUSE be preprocessed in the same way as the model’s training data:
#       Just use preprocess_input()
# NOTE: preprocess_input() requires pixels' value 0 to 255
inputs = keras.applications.resnet50.preprocess_input(images)

# 6.4. Predict
# NOTE: Keep in mind that the model had to choose from among 1,000 classes!
Y_proba = model.predict(inputs)
k=3
top_k_predictions = keras.applications.resnet50.decode_predictions(Y_proba, top=k)
for i in range(len(images)):
    if 10:
        plt.imshow(images[i]/80) # processed images do NOT have pixel values of 0-255
        plt.title(file_names[i])
        plt.show()
    print("File {}".format(file_names[i]))
    for class_id, name, y_proba in top_k_predictions[i]:
        #print("  {:12s} (id:{}): {:.1f}%".format(name,class_id,  y_proba * 100))
        print("   {}: {:.1f}%".format(name, y_proba*100))
    print()

#endregion


# In[7]:  TRANSFER LEARNING WITH PRETRAINED MODELs from tf.keras.efficientnet
#region 
# 6.1. Load data and split 
import tensorflow_datasets as tfds
dataset, info = tfds.load("tf_flowers", as_supervised=True, with_info=True)
#data = tfds.load('oxford_flowers102') smallnorb
class_names = info.features["label"].names
n_classes = info.features["label"].num_classes
dataset_size = info.splits["train"].num_examples
test_set_raw, valid_set_raw, train_set_raw = tfds.load(
    "tf_flowers",
    split=["train[:10%]", "train[10%:25%]", "train[25%:]"],
    as_supervised=True)
if 1: 
    plt.figure(figsize=(12, 15))
    index = 0
    for image, label in train_set_raw.take(6):
        index += 1
        plt.subplot(3, 2, index)
        plt.imshow(image)
        plt.title("Class: {}".format(class_names[label]), fontsize=25)
        #plt.axis("off")

#%% 6.2. Augment and preprocess images
# NOTE: efficientnet-b3 takes 300x300 images. Source: https://kobiso.github.io/Computer-Vision-Leaderboard/imagenet.html
def central_crop(image):
    #image = cv2.imread('images\Meo_2.jpg')
    shape = tf.shape(image)
    min_dim = tf.reduce_min([shape[0], shape[1]])
    top_crop = (shape[0] - min_dim) // 4
    bottom_crop = shape[0] - top_crop
    left_crop = (shape[1] - min_dim) // 4
    right_crop = shape[1] - left_crop
    return image[top_crop:bottom_crop, left_crop:right_crop]

def random_crop(image, kept_percentage=90): # crop out 10% (default)
    shape = tf.shape(image)
    min_dim = tf.reduce_min([shape[0], shape[1]]) * kept_percentage // 100
    return tf.image.random_crop(image, [min_dim, min_dim, 3])

def preprocess(image, label, x_size=300, y_size=300, rand_crop=False):
    if rand_crop:
        cropped_image = random_crop(image)
        cropped_image = tf.image.random_flip_left_right(cropped_image)
    else:
        cropped_image = central_crop(image)
    resized_image = tf.image.resize_with_pad(cropped_image, x_size, y_size)
    
    # Add other operations here: eg., 
    # tf.image.random_flip_left_right()  # flip images

    final_image = tf.keras.applications.efficientnet.preprocess_input(resized_image)
    return final_image, label

from functools import partial
batch_size = 32
train_set = train_set_raw.shuffle(buffer_size=1000).repeat() 
# NOTE: 
#   1. repeat(None): infinitely duplicate the data. (repeat(2): duplicate 2 times). To see, try: dem=0; for i in train_set: dem+=1; print(dem)
#   2. Unfortunately, NO way to get shape of tf.dataset except using loop (as NOTE 1.). Or see 'info'.num_examples of the dataset
train_set = train_set.map(partial(preprocess, rand_crop=True)).batch(batch_size).prefetch(1)
valid_set = valid_set_raw.repeat().map(preprocess).batch(batch_size).prefetch(1)
test_set = test_set_raw.map(preprocess).batch(batch_size).prefetch(1)
# NOTE: Each batch includes: 1. Images, 2. List of labels.
if 10: # Plot some samples
    for batch in train_set.take(1):
        images, labels = batch[0], batch[1]
        plt.figure(figsize=(15, 30))
        for i in range(8): #len(labels)
            plt.subplot(4, 2, i+1)
            plt.imshow(images[i]/255)
            plt.axis('off')
            plt.title('Label: '+class_names[labels[i]], fontsize=30)
        plt.show()

#%% 6.3. Load EfficientNet B3
# Load model:
base_model = keras.applications.efficientnet.EfficientNetB3(weights="imagenet",
                include_top=False) # NOT include the FC layers
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation="softmax")(avg)
model = keras.models.Model(inputs=base_model.input, outputs=output)
# Display model: 
for index, layer in enumerate(model.layers):
    print(index, layer.name)
#model.summary()


#%% 6.4. TRANSFER LEARNING
# NOTE: Recall: TRANSFER LEARNING STEPS
#   0. Load pretrained model (usually trained with ImageNet). Add FC layers (to fit your data labels).
#   1. Freeze base-model layers: to train only FC layers. After a few epochs, its validation accuracy stops making much progress => FC layers are now pretty well trained.
#   3. Unfreeze and train all layers (or just the top ones). NOTE: Use a much lower learning rate to avoid damaging the pretrained weights.

# STEP 1: Freeze the base_model layers: to train only the FC layers
# CAUTION: takes a while to finish! ==> Use GPU or try using Google Colab.
new_training = 0
if new_training:
    for layer in base_model.layers:
        layer.trainable = False

    optimizer = keras.optimizers.Nadam(lr=0.2)
    # optimizer = keras.optimizers.SGD(lr=0.2, momentum=0.9, decay=0.01)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    #with tf.device('/cpu'):
    history = model.fit(train_set,
                        steps_per_epoch=int(0.75*dataset_size / batch_size),
                        validation_data=valid_set,
                        validation_steps=int(0.15*dataset_size / batch_size),
                        epochs=4)
    # NOTE: steps_per_epoch (typically)= dataset_size // batch_size
    #       However, you can change this number to "trick" the trainer, e.g., to update the learning rate using ReduceLROnPlateau() callback, or just to stop training sooner.
    #       Or, in case of infinite training data (.repeat(None)), we MUST specify this for the training to stop.
    model.save('models/effB3_trainFClayers.h5')
    test_loss, test_acc = model.evaluate(test_set)
    import joblib
    joblib.dump(test_acc, 'models/effB3_trainFClayers_testAccuracy')
else:
    model = keras.models.load_model('models/effB3_trainFClayers.h5')
    test_acc = joblib.load('models/effB3_trainFClayers_testAccuracy')
print('Test accuracy:',test_acc)
# Try prediction:
if 10: 
    plt.figure(figsize=(12, 80))
    index = 0
    test_set_raw = test_set_raw.shuffle(buffer_size=50)
    for image, label in test_set_raw.take(30):
        index += 1
        plt.subplot(15, 2, index)
        plt.imshow(image)

        test_img, label = preprocess(image, label)
        with tf.device('/cpu'):
            prediction =  model(test_img, training=False) # NOTE: model.predict() is designed for performance in large scale inputs. For small amount of inputs that fit in one batch, directly using __call__ is recommended for faster execution, e.g., model(x), or model(x, training=False) 
        prediction_lbl = np.argmax(prediction.numpy())
        prediction_score = np.max(prediction.numpy())

        plt.title("Label: {} \nTop-1 predict: {} ({}%)".format(class_names[label],class_names[prediction_lbl], round(prediction_score*100,1)), fontsize=20)
        plt.axis("off")


#%% STEP 2: Unfreeze and train all layers (or just the top ones)
# CAUTION: takes a while to finish! ==> Use GPU or try using Google Colab.
class OneCycleScheduler(keras.callbacks.Callback):
    # Source: (Géron, 2019)
    def __init__(self, iterations, max_rate, start_rate=None,
                 last_iterations=None, last_rate=None):
        self.iterations = iterations
        self.max_rate = max_rate
        self.start_rate = start_rate or max_rate / 10
        self.last_iterations = last_iterations or iterations // 10 + 1
        self.half_iteration = (iterations - self.last_iterations) // 2
        self.last_rate = last_rate or self.start_rate / 1000
        self.iteration = 0
    def _interpolate(self, iter1, iter2, rate1, rate2):
        return ((rate2 - rate1) * (self.iteration - iter1)
                / (iter2 - iter1) + rate1)
    def on_batch_begin(self, batch, logs):
        K = keras.backend
        if self.iteration < self.half_iteration:
            rate = self._interpolate(0, self.half_iteration, self.start_rate, self.max_rate)
        elif self.iteration < 2 * self.half_iteration:
            rate = self._interpolate(self.half_iteration, 2 * self.half_iteration,
                                     self.max_rate, self.start_rate)
        else:
            rate = self._interpolate(2 * self.half_iteration, self.iterations,
                                     self.start_rate, self.last_rate)
            rate = max(rate, self.last_rate)
        self.iteration += 1
        K.set_value(self.model.optimizer.lr, rate)

new_training = 0
if new_training:
    batch_size = 32 
    good_lr = 0.1
    init_lr = good_lr/10

    model = keras.models.load_model('models/effB3_trainFClayers.h5')
    for layer in model.layers[-10:]:
        layer.trainable = True
    optimizer = keras.optimizers.Nadam(learning_rate=init_lr)
    # optimizer = keras.optimizers.SGD(learning_rate=0.03, momentum=0.9, nesterov=True, decay=0.001)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    checkpoint_saver = keras.callbacks.ModelCheckpoint('models/effB3_trainMoreLayers.h5', save_best_only=True)
    early_stopper = keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)
    #performance_sched = keras.callbacks.ReduceLROnPlateau(patience=3)

    n_epochs = 20
    n_iters = int(dataset_size / batch_size) * n_epochs
    onecycle = OneCycleScheduler(n_iters, max_rate=good_lr)

    #with tf.device('/cpu'):
    history = model.fit(train_set,
                        steps_per_epoch=int(0.75 * dataset_size / batch_size),
                        validation_data=valid_set,
                        validation_steps=int(0.15 * dataset_size / batch_size),
                        epochs=n_epochs,
                        callbacks=[checkpoint_saver,early_stopper,onecycle])
    
    test_loss, test_acc = model.evaluate(test_set)
    joblib.dump(test_acc, 'models/effB3_trainMoreLayers_testAccuracy')
else:
    model = keras.models.load_model('models/effB3_trainMoreLayers.h5')
    test_acc = joblib.load('models/effB3_trainMoreLayers_testAccuracy')
print('Test accuracy:',test_acc)
# Try prediction:
if 10: 
    plt.figure(figsize=(12, 80))
    index = 0
    test_set_raw = test_set_raw.shuffle(buffer_size=50)
    for image, label in test_set_raw.take(30):
        index += 1
        plt.subplot(15, 2, index)
        plt.imshow(image)

        test_img, label = preprocess(image, label)
        #with tf.device('/cpu'):
        prediction =  model(test_img, training=False) 
        prediction_lbl = np.argmax(prediction.numpy())
        prediction_score = np.max(prediction.numpy())

        plt.title("Label: {} \nTop-1 predict: {} ({}%)".format(class_names[label],class_names[prediction_lbl], round(prediction_score*100,1)), fontsize=20)
        plt.axis("off")

#endregion


''' WEEK 11 '''

# In[8]: TRY CREATING AN OBJECT DETECTOR USING A PRETRAINED CNN
#region
# 8.0. Try create your image annotations and load them
def plot_bbox(plt, ymin, xmin, ymax, xmax, img_height=1, img_width=1, boxcolor='b', boxedgewidth=3, label_str='', fontsize=8, text_background='y'):
    '''Plot bounding box.
    plt: plot handle. Gotten by: import matplotlib.pyplot as plt 
    ymin, xmin, ymax, xmax: coordinates of the box. NOTE: if use normalized coordinates, MUST specify img_height and img_width.
    img_height, img_width: dimension of image
    label_str: string of label to show
    '''    
    (ymin, ymax) =  (int(y*img_height) for y in (ymin, ymax)) # convert to absolute coordinates
    (xmin, xmax) =  (int(x*img_width) for x in (xmin, xmax))
    import matplotlib.patches as patches
    rect = patches.Rectangle((xmin, ymin), width=xmax-xmin, height=ymax-ymin, linewidth=boxedgewidth, edgecolor=boxcolor,facecolor='none')
    ax = plt.gca()
    ax.add_patch(rect)
    plt.text(xmin, ymin, label_str, fontsize=fontsize, backgroundcolor=text_background)        

# NOTE: COCO format: https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch 
file_handle = open(r'images\cats_and_stuff\via_project_5May2021_21h26m_coco.json')
import json
json_data = json.load(file_handle) # json_data: a dictionary the same as json file
class_names = []
for cat in json_data['categories']:
    class_names.append(cat['name'])    
for image in json_data['images']:
    file_name = image['file_name']
    image_id = image['id']
    img = cv2.imread(r'images\cats_and_stuff\\'+file_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    list_bbox = []
    for annotation in json_data['annotations']:
        if annotation['image_id'] == image_id:
            bbox = annotation['bbox']
            (xmin, ymin, width, height) = bbox
            ymax, xmax = ymin+height, xmin+width
            category_id = annotation['category_id']
            label_str = class_names[category_id-1]
            plot_bbox(plt, ymin, xmin, ymax, xmax, label_str=label_str)
            list_bbox.append(bbox)            
    plt.show()  
file_handle.close() # Don't forget to close the file.

#%% 8.1. Load data PASCAL Visual Object Classes Challenge,https://www.tensorflow.org/datasets/catalog/voc 
# NOTE: 
#   Test set of VOC2012 does not contain annotations.
#   Bounding box: [ymin, xmin, ymax, xmax] https://www.tensorflow.org/datasets/api_docs/python/tfds/features/BBoxFeature 
#   Normalized bounding box: coordinates are rescaled to the 0-1 range by dividing by the image width and height, https://github.com/ultralytics/yolov3/issues/475
import tensorflow_datasets as tfds 
data, info = tfds.load("voc", with_info=True) 
train_set = data['train']
val_set = data['validation']
#test_set = data['test'] # NOTE: test set of VOC2012 does not contain annotations.
train_size = info.splits['train'].num_examples
test_size = info.splits['test'].num_examples
val_size = info.splits['validation'].num_examples
no_of_classes = info.features['labels'].num_classes
class_names = info.features["labels"].names
print(info)

# Plot some samples:
train_set = train_set.shuffle(buffer_size=100)
for item in train_set.take(3):
    img = item['image']
    height, width, depth = img.shape    
    plt.imshow(img)
    
    list_bbox = item['objects']['bbox']
    list_lbl = item['objects']['label'] 
    list_is_diff = item['objects']['is_difficult'] 
    list_is_trunc = item['objects']['is_truncated'] 
    for i in range(len(list_bbox)):
        box = list_bbox[i]
        (ymin, xmin, ymax, xmax) = box # normalized coordinates        
        
        label_id = list_lbl[i]
        label = class_names[label_id]
        is_diff = 'D'+str(int(list_is_diff[i]))
        is_trunc = 'T'+str(int(list_is_trunc[i]))
        #label_str = label+'('+is_diff+','+is_trunc+')' 
        label_str = label 

        plot_bbox(plt, ymin, xmin, ymax, xmax, img_height=height, img_width=width, label_str=label_str)
    plt.show()


#%% 8.2. Load a pretrained model and add output layers
base_model = keras.applications.efficientnet.EfficientNetB3(weights="imagenet",
                include_top=False) # NOT include the FC layers
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
class_output = keras.layers.Dense(no_of_classes, activation="softmax", name='class_output')(avg)
loc_output = keras.layers.Dense(4, activation="sigmoid", name='loc_output')(avg) 
# NOTE: 
#   + This net gives ONLY 1 box (ie, 1 object) per image. => Keep ONLY 1 object bbox per image in the training images
#   + activation="sigmoid": b/c of normalized coordinates (0-1)
model = keras.models.Model(inputs=base_model.input, 
                           outputs=[class_output, loc_output])
# NOTE: With this model, for the fit(x=,y=(,)) to work, 
#       each training item MUST be in the form: (image, (label, bbox))
model.compile(loss=["sparse_categorical_crossentropy", "mse"],
              loss_weights=[0.6, 0.4], # depends on what you care most about
              optimizer=keras.optimizers.Nadam(), 
              metrics=["accuracy"] )
# NOTE: How does tf compute 'accuracy' for dense outputs? It depends on your loss and output shape. For example, it rounds the output and counts the matches. See a good explaination here: https://stackoverflow.com/questions/55828344/how-does-tensorflow-calculate-the-accuracy-of-model

#%% 8.3. Prepare and pack data in the same form of model's input, output
# NOTE: for above model, each training item MUST be in the form: (image, (label, bbox))
def prepare_pack_data(image, label, bbox, x_size=300, y_size=300):
    # Preprocess for Efficientnet-B3: x_size=300, y_size=300
    #resized_image = tf.image.resize_with_pad(image, x_size, y_size)
    resized_image = tf.image.resize(image, (x_size, y_size))
    # Add augmentation operations here: eg., crop, flip
    # tf.image.random_flip_left_right()  # flip images
    image = tf.keras.applications.efficientnet.preprocess_input(resized_image)
    return image, (label, bbox)

batch_size = 5
prepared_train_set = train_set.repeat().shuffle(500)
# NOTE: Above net gives ONLY 1 box (ie, 1 object) per image. => Keep ONLY 1 object bbox per image
prepared_train_set = prepared_train_set.take(-1).map(lambda item: prepare_pack_data(item['image'], item['objects']['label'][0], item['objects']['bbox'][0])) # NOTE: Keep ONLY 1 object bbox per image
prepared_train_set = prepared_train_set.batch(batch_size).prefetch(1) 

if 10: # Plot for reviewing purpose
    for image_batch, (label_batch, bbox_batch) in prepared_train_set.take(1): 
        for image, label, bbox in zip(image_batch, label_batch, bbox_batch):
            print('label:', label)
            plt.imshow(image/255)
            (ymin, xmin, ymax, xmax) = bbox
            h, w, d = image.shape
            lbl_str = class_names[label]
            plot_bbox(plt, ymin, xmin, ymax, xmax, h, w, label_str=lbl_str)
            plt.show()
 
#%% 8.4. Train the model
new_training = 0
if new_training:
    early_stopper = keras.callbacks.EarlyStopping(monitor='loc_output_accuracy',patience=3)
    model_checkpoint = keras.callbacks.ModelCheckpoint('models/try_eff_B3_detection.h5',monitor='loc_output_accuracy',save_best_only=True)
    model.fit(prepared_train_set, batch_size=batch_size, 
          steps_per_epoch=10, epochs=30, 
          callbacks = [early_stopper,model_checkpoint])
model = keras.models.load_model('models/try_eff_B3_detection.h5')

#%% 8.5. Try detection
# NOTE: Use model(X) to predict small number of data. Use model.predict() for many data.
for image_batch, (label_batch, bbox_batch) in prepared_train_set.take(1): 
    for image, label, bbox in zip(image_batch, label_batch, bbox_batch):
        # True label:
        plt.subplot(1,2,1)
        plt.imshow(image/255)
        (ymin, xmin, ymax, xmax) = bbox
        h, w, d = image.shape
        lbl_str = class_names[label]
        plot_bbox(plt, ymin, xmin, ymax, xmax, h, w, label_str=lbl_str)
        plt.title('True label:')

        # Prediction:
        plt.subplot(1,2,2)
        plt.imshow(image/255)
        proba, bbox = model(image)
        (ymin, xmin, ymax, xmax) = bbox[0]
        h, w, d = image.shape
        label = np.argmax(proba)
        lbl_str = class_names[label]
        plot_bbox(plt, ymin, xmin, ymax, xmax, h, w, label_str=lbl_str)
        plt.title('Prediction:')
        plt.show()

#endregion


# In[9]: FULLY CONVOLUTIONAL NETS 
#region
#%% 9.1. Dense vs Conv2D
print('>>>>> MODEL 1: Dense output layer:')
test = keras.Sequential([ 
    keras.layers.Input(shape=(150,150,3)),
    keras.layers.Conv2D(filters=100, kernel_size=(7,7),strides=10),
    keras.layers.Flatten(),
    keras.layers.Dense(200)
    ])
test.summary()

print('\n\n>>>>> MODEL 2: Conv2D output layer has the same #param as the Dense layer:')
test2 = keras.Sequential([ 
    keras.layers.Input(shape=(150,150,3)),
    keras.layers.Conv2D(filters=100, kernel_size=(7,7), strides=10),
    keras.layers.Conv2D(filters=200, kernel_size=(15,15), strides=1),
    ])
test2.summary()

print('\n\n>>>>> MODEL 3: Change input image size, still get the same #param (but different ouput shape of the Conv2D):')
test3 = keras.Sequential([ 
    keras.layers.Input(shape=(300,300,3)),
    keras.layers.Conv2D(filters=100, kernel_size=(7,7), strides=10),
    keras.layers.Conv2D(filters=200, kernel_size=(15,15), strides=1),
    ])
test3.summary()


#%% 9.2. Compute mAP
def maximum_precisions(precisions):
    return np.flip(np.maximum.accumulate(np.flip(precisions)))

recalls = np.linspace(0, 1, 11)
precisions = [0.91, 0.94, 0.96, 0.94, 0.95, 0.92, 0.80, 0.60, 0.45, 0.20, 0.10]
max_precisions = maximum_precisions(precisions)
AP = max_precisions.mean()
if 1:
    plt.plot(recalls, precisions, "ro--", label="Precision")
    plt.plot(recalls, max_precisions, "bo-", label="Max Precision")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.plot([0, 1], [AP, AP], "g:", linewidth=3, label="AP")
    plt.grid(True)
    plt.axis([0, 1, 0, 1])
    plt.legend(loc="lower center", fontsize=14)
    plt.show()

#%% 9.3. Transpose convolutions (Sementic segmentation)
# Load image:
img_BGR = cv2.imread(r'images/Meo_2.jpg')
img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
img = tf.image.resize(img, (200,250), antialias=True) 
X = img[np.newaxis, :, :, :]
plt.imshow(img/255)
plt.title('Original image, size:' + str(X.shape))
plt.show()

# Upsampling the image:
conv_transpose = keras.layers.Conv2DTranspose(filters=3, kernel_size=3, strides=2, padding="VALID") # NOTE: larger stride, larger output
output = conv_transpose(X)
def normalize(X): # to make sure pixel value is in [0, 1]
    return (X - tf.reduce_min(X)) / (tf.reduce_max(X) - tf.reduce_min(X))
output = normalize(output)
plt.imshow(output[0])
plt.title('Upscaled image, size:' + str(output.shape))
plt.show()

#endregion

