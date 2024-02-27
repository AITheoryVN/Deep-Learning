''' 
Main reference: Chapter 13 in (Géron, 2019)
Last review: March 2022
'''

# In[0]: IMPORTS AND SETTINGS
#region
import sys
from numpy.core import shape_base
from tensorflow._api.v2 import data
from tensorflow.keras import callbacks
from wrapt.wrappers import patch_function_wrapper
assert sys.version_info >= (3, 5) # Python ≥3.5 is required
import sklearn
assert sklearn.__version__ >= "0.20" # Scikit-Learn ≥0.20 is required
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0" # TensorFlow ≥2.0 is required
import numpy as np
import os
np.random.seed(42)
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
#endregion


''' WEEK 07 '''

# In[1]: TF DATASET
#region
# 1.1. Load data and convert to tf dataset
sample_data = [11,22,33,44,55,66,77,88]
dataset = tf.data.Dataset.from_tensor_slices(sample_data) # convert data into tf data
print('Dataset:')
#dataset = dataset.take(5) # take first 5 samples
for item in dataset:
    print(item)

# 1.2. Filter data
dataset = dataset.filter(lambda sample: sample>30)
print('\nFilter samples >30:')
for item in dataset:
    print(item)
    #print(item.numpy()) # get the content of a tensor

# 1.3. Data transformations. NOTE: these functions are NOT in-place, hence require re-assignment
#dataset = dataset.repeat(3) # duplicate data
#dataset = dataset.batch(4) # group data. drop_remainder=True: drop the last batch
dataset = dataset.repeat(3).batch(5) # do above 2 transformations at once
print('\nTransformed dataset (repeat(3).batch(5)):')
for item in dataset:
    print(item)

# 1.4. Custom transformation on EACH SAMPLE
dataset = dataset.unbatch() # ungroup data
# Syntax 1 (only for simple transform)
# NOTE: num_parallel_calls = #threads
dataset = dataset.map(lambda sample: sample*10 if sample<60 else sample, num_parallel_calls=4)
# Syntax 2 (CLEAREST & most flexible) 
def my_func_sample(x):
    if x<60:
        x = x*10
    return x
dataset = dataset.map(lambda sample: my_func_sample(sample), num_parallel_calls=4)
# Syntax 3 (short form of syntax 2)
dataset = dataset.map(my_func_sample, num_parallel_calls=4)
print('\nCustom transform EACH SAMPLE (x10 if <60):')
for item in dataset:
    print(item)

# 1.5. Custom transformation on WHOLE DATASET
# Syntax 1 (simple transform)
max_val = 1000
dataset = dataset.apply(lambda datset: datset.filter(lambda sample: sample<max_val))

#import tensorflow_transform as tft
#dataset = dataset.apply(lambda datset: datset - tft.mean(datset))

# Syntax 2 (CLEAREST) 
def my_func(ds):
    ds = ds.filter(lambda sample: sample<max_val)
    new_ds = ds.map(lambda sample: sample/10)
    return new_ds 
dataset = dataset.apply(lambda datset: my_func(datset))
# Syntax 3 (short form of syntax 2)
dataset = dataset.apply(my_func)
print('\nCustom transform WHOLE DATASET (/10 if <max_val):')
for item in dataset:
    print(item)

#%% 1.6. Randomly shuffle the dataset
# NOTE: shuffle dataset helps to make sure your training data are iid 
#       (REQUIRED in training using gradient descent)
# 1.6.1. For small dataset
# INFO: shuffle() method works by getting N items of the dataset (N='buffer_size') 
#       into RAM each time. Then it randomly draws samples 
#       from this buffer and replaces the drawn ones with new samples from the dataset.
# NOTE: MUST set 'buffer_size' so that the buffet doesn't exceed RAM capacity.
dataset = dataset.shuffle(buffer_size=3, seed=42)
print('\nRandom samples:')
for item in dataset:
    print(item)

#%% 1.6.2. For LARGE dataset
# NOTE: shuffle() alone CAN'T shuffle well large dataset since the buffer size is relatively small compared to the dataset's size.
# SOLUTION: 
#   1. Split dataset into muptiple files.
#   2. Read these files at once to draw random samples from ALL parts of the dataset.
from sklearn.datasets import fetch_california_housing
# Demo on California housing dataset
# INFO: 20640 samples, 8 features, label values: 0.15 - 5.
#       https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html 
housing = fetch_california_housing()
print("Cali housing dataset size:",housing.data.shape)
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target.reshape(-1, 1), random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

# STEP 1: Save data to multiple files 
def save_to_multiple_csv_files(data, name_prefix, header=None, n_parts=10):
    housing_dir = os.path.join("datasets", "housing")
    os.makedirs(housing_dir, exist_ok=True)
    path_format = os.path.join(housing_dir, "my_{}_{:02d}.csv")

    filepaths = []
    m = len(data)
    for file_idx, row_indices in enumerate(np.array_split(np.arange(m), n_parts)):
        part_csv = path_format.format(name_prefix, file_idx)
        filepaths.append(part_csv)
        with open(part_csv, "wt", encoding="utf-8") as f:
            if header is not None:
                f.write(header)
                f.write("\n")
            for row_idx in row_indices:
                f.write(",".join([repr(col) for col in data[row_idx]]))
                f.write("\n")
    return filepaths
train_data = np.c_[X_train, y_train]
valid_data = np.c_[X_valid, y_valid]
test_data = np.c_[X_test, y_test]
header_cols = housing.feature_names + ["MedianHouseValue"]
header = ",".join(header_cols)
train_filepaths = save_to_multiple_csv_files(train_data, "train", header, n_parts=20)
valid_filepaths = save_to_multiple_csv_files(valid_data, "valid", header, n_parts=10)
test_filepaths = save_to_multiple_csv_files(test_data, "test", header, n_parts=10)
print('\nDone writing files. Training file paths:',train_filepaths)

# STEP 2: Read created files at once 
# Read some first samples of a file (JUST TO CHECK)
import pandas as pd
print('\nSome first samples:')
print(pd.read_csv(train_filepaths[0]).head())

# Create a dataset containing file paths in RANDOM ORDER, using list_files() 
print('\nFile paths (in RANDOM ORDER):')
filepath_dataset = tf.data.Dataset.list_files(train_filepaths, shuffle=True, seed=42)
for filepath in filepath_dataset:
    print(filepath)

# Create a dataset containing data from MULTIPLE FILES, using interleave()
# INFO: interleave() creates a dataset from N (N=cycle_length) RANDOM files (with names in 'filepath_dataset'). 
#       When you get data from this dataset, you will get first rows of these N random files (one row/file each time).
#       When you get all rows from these N files, the other N files (from 'filepath_dataset') will be generated.
# NOTE: files (with names in 'filepath_dataset') SHOULD have the identical length,
#       otherwise the ends of the longer files won't be gotten.      
N_files_1_read = 5
dataset = filepath_dataset.interleave(
    lambda filepath: tf.data.TextLineDataset(filepath).skip(1), # skip(1): the header row
    cycle_length=N_files_1_read)
print('\nSome first samples of the SHUFFLED dataset:')
for item in dataset.take(7):
    print(item)
    #print(item.numpy())

#%% 1.7. Converts CSV records to tensors using decode_csv()
# INFO: 
#   + decode_csv() converts CSV records to tf tensors.
#   + A CSV record is a string with commas and NO space.
#sample_record = '11,22,33,44,55' # with NO missing values
sample_record = ',22,,,55' # with MISSING values
# NOTE on creating DEFAULT VALUES:
#   + No. of default values MUST match exactly no. of fields in the records.
#   + Empty default values (eg. tf.constant([])) mean the fields are REQUIRED (no missing allowed).
default_values=[0.101, np.nan, tf.constant(np.nan, dtype=tf.float64), "Hello", tf.constant([])] 
#default_values=[0.101, np.nan, "Hello", tf.constant([])] # NOT enough values => ERROR
processed_record = tf.io.decode_csv(sample_record, default_values)
processed_record

#%% 1.8. Preprocessing data
# 1.8.1. Processing 1 record
@tf.function # Convert below function to tf function (FASTER than regular python function)
def preprocess(line, no_of_features=1):
    default_val = [0.]*no_of_features + [tf.constant([], dtype=tf.float32)] # last field is the label (CAN'T be missing)
    fields = tf.io.decode_csv(line, record_defaults=default_val)
    x = tf.stack(fields[:-1]) # tf.stack(): merges elements into 1 array
    y = tf.stack(fields[-1:])
    # Do other preprocessing here...
    return x, y
no_of_features = X_train.shape[-1]
sample_record = b'4.6477,38,,0.911864406779661,745.0,2.5254237288135593,32.64,-117.07,1.504' # b: string of byte literals
(x,y) = preprocess(sample_record, no_of_features)
print('\nSample record:\nx =',x,'\ny =',y)

# 1.8.2. Preprocessing a dataset
# >> See slides  
def csv_reader_dataset(filepaths, no_of_features, line_skip=1, no_files_1_read=5,
                       num_threads=1, shuffle_buffer_size=10000, batch_size=32):
    # Create a dataset of file paths:
    dataset = tf.data.Dataset.list_files(filepaths, shuffle=True)
    # Create a dataset of shuffled samples from files (in the 'filepaths'):
    dataset = dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(line_skip),
        cycle_length=no_files_1_read, num_parallel_calls=num_threads)
    # Cache the data in RAM for speed:
    dataset = dataset.cache() # NOTE: dataset size MUST <= RAM capacity
    # Shuffle records 1 more time:
    dataset = dataset.shuffle(shuffle_buffer_size)
    # Preprocess records:
    dataset = dataset.map(lambda line: preprocess(line,no_of_features), num_parallel_calls=num_threads)
    # Group records into batches:
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(1) # NOTE: prefetch(): can increase running speed significantly, by preparing the data AHEAD of calling.
dataset = csv_reader_dataset(train_filepaths, no_of_features, batch_size=2)
print('\n\nProcessed training data:')
for record in dataset.take(3):
    print('\n',record)
