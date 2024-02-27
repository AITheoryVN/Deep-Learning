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
# E.g., when you need to compute mean, max...
# Syntax 1 (simple transform)
max_val = 1000
#dataset = dataset.apply(lambda datset: datset.filter(lambda sample: sample<max_val))

# Syntax 2 (CLEAREST) 
def my_func(ds):
    ds = ds.filter(lambda sample: sample<max_val)
    
    # Compute mean
    sum = ds.reduce(np.int32(0), lambda old_state, sample: old_state+sample).numpy()
    no_of_samples = ds.reduce(np.int32 (0), lambda old_state, _: old_state+1).numpy()
    mean = sum/no_of_samples

    new_ds = ds.map(lambda sample: sample - mean)    
    return new_ds 
dataset = dataset.apply(lambda datset: my_func(datset))

# Syntax 3 (short form of syntax 2)
#dataset = dataset.apply(my_func)
print('\nCustom transform WHOLE DATASET:')
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


#%% 1.9. [Exercise] PUTTING THINGS TOGETHER: Load data and train a NN
 

#%% 1.10. [qAdvanced] WRITE YOUR OWN TRAINING LOOP FUNCTION
# Original code: cell 40 in '13_loading_and_preprocessing_data.ipynb'
# NOTE: requires knowledge in Chap 12
@tf.function
def train(model, n_epochs, train_filepaths, no_of_features, optimizer, loss_fn, 
          batch_size=32, no_files_1_read=5, num_threads=1, shuffle_buffer_size=10000):
    train_set = csv_reader_dataset(train_filepaths, no_of_features, no_files_1_read=no_files_1_read,
                                   num_threads=num_threads, shuffle_buffer_size=shuffle_buffer_size, batch_size=batch_size)
    for epoch in tf.range(n_epochs):
        iter = 0
        for X_batch, y_batch in train_set:
            iter += 1
            with tf.GradientTape() as tape:
                y_pred = model(X_batch)
                main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
                loss = tf.add_n([main_loss] + model.losses)
                tf.print("\rEpoch ", epoch+1, "/", n_epochs,', iter ',iter,': loss = ',loss,sep='')
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
n_epochs=2
no_of_features = X_train.shape[-1] # = 8
optimizer = keras.optimizers.Nadam(lr=0.01)
loss_fn = keras.losses.mean_squared_error
train(model, n_epochs, train_filepaths, no_of_features, optimizer, loss_fn)
#endregion


# In[2]: TFRECORD
# >> See slides
#region
# 2.1. Create TFRecord files
with tf.io.TFRecordWriter("datasets/my_data1.tfrecord") as f:
    f.write(b"First record: What's up man?") # binary literals (only ASCII char)
    f.write(b"Record 2: Just some fun coding.")
with tf.io.TFRecordWriter("datasets/my_data2.tfrecord") as f:
    f.write(u"Đại học Sư Phạm Kỹ Thuật TP. HCM ") # Unicode literals 
    f.write("Khoa CNTT") # Unicode is default in Python 3)

# 2.2. Read TFRecord files    
filepaths = ["datasets/my_data1.tfrecord", "datasets/my_data2.tfrecord"]
dataset = tf.data.TFRecordDataset(filepaths)
print('\nSample TFRecord records:')
for item in dataset: 
    #print(item) 
    #print(item.numpy()) # get content of a tensor
    print(item.numpy().decode('UTF-8'))

# 2.3. Shuffle TFRecord files' records
# INFO: just as before
# For small files: use shuffle()
dataset = dataset.shuffle(buffer_size=5)
print('\nShuffled records (small files):')
for item in dataset: 
    print('\t',item.numpy().decode('UTF-8'))
# For LARGE files: store data into multiple files and interleave their records
# STEP 1: store data into multiple files
n_files = 3
filepaths = ["datasets/my_file_{}.tfrecord".format(i) for i in range(n_files)]
for file_id, filepath in enumerate(filepaths):
    with tf.io.TFRecordWriter(filepath) as f:
        for rec in ['A','B','C','D']:
            f.write("Record {} (file {}).".format(rec, file_id).encode("utf-8"))
# STEP 2: interleave their records
# NOTE: MUST set "num_parallel_reads">1 to read more than 1 file each time.
dataset = tf.data.TFRecordDataset(filepaths, num_parallel_reads=3)
dataset = dataset.shuffle(buffer_size=5)
print('\nShuffled records (LARGE files):')
for item in dataset: 
    print('\t',item.numpy().decode('UTF-8'))

# 2.4. Compressed TFrecord files
# Create compressed TFrecord files
options = tf.io.TFRecordOptions(compression_type="GZIP")
with tf.io.TFRecordWriter("datasets/my_compressed_0.tfrecord", options) as f:
    for rec in ['A','B','C','D']:
        f.write("Record {} (file {}).".format(rec, 0).encode("utf-8"))
# Read compressed TFrecord files
dataset = tf.data.TFRecordDataset(["datasets/my_compressed_0.tfrecord"],
                                  compression_type="GZIP")
print('\nCompressed records:')
for item in dataset:
    print(item.numpy())

#endregion


# In[21]: [qAdvanced] PROTOBUF
# CAUTION: Again, if it's OK with other data formats (eg, csv, jpg...), then NO NEED to use TF Record
#region
# 21.1. To make your custom protobufs:
#   0. Download protoc-....zip from https://github.com/protocolbuffers/protobuf/releases
#   1. Create definitions (stored in a .proto file)
#   2. Compile the protobuf definitions using protoc (in protoc-.../bin)
#   3. Import and use them as normal Python classes

# 1. Create definitions in a .proto file
'''
syntax = "proto3";
message Person {
  string name = 1;
  int32 id = 2;
  repeated string email = 3;
}
'''
# 2. Compile the .proto file
# cmd:
#   protoc qPerson.proto --python_out=.
# (using protoc in protoc-.../bin. Add ENV PATH if necessary)

# 3. Import and use them as normal Python classes
from protobuf_files.qPerson_pb2 import Person
person1 = Person(name="Thien", email=["a@b.com"]) 
print(person1.name)
person1.email.append("c@d.com")
person1.email.append("e@d.com")
person1.email.append("f@d.com")
print(person1.email)
s = person1.SerializeToString()  # serialize. NOT a TF function. Ready to save or transmit via netword 
person2 = Person()  # create a new Person
person2.ParseFromString(s) 
person1 == person2

# %% 21.2. EXAMPLE protobuf
# INFO: Example is pre-defined protobuf that represents a SAMPLE in a dataset.
# NOTE: 'features', 'feature' are COMPULSORY keywords in Example protobuf
import tensorflow as tf
from tensorflow.train import BytesList, FloatList, Int64List #just lists of bytes, float...
from tensorflow.train import Feature, Features, Example
person3 = Example(
    features = Features(
        feature={
            "name": Feature(bytes_list=BytesList(value=[b"Tri"])),
            "id": Feature(int64_list=Int64List(value=[1133])),
            "emails": Feature(bytes_list=BytesList(value=[b"a@b.com", b"c@d.com"]))
        }))
print(person3.features.feature['id'])
person4 = Example(
    features = Features(
        feature={
            "name": Feature(bytes_list=BytesList(value=[b"Dung"])),
            "id": Feature(int64_list=Int64List(value=[1144])),
            "emails": Feature(bytes_list=BytesList(value=[b"a@b.com", b"c@d.com"]))
        }))

# Serialize Examples to a file
with tf.io.TFRecordWriter("datasets/person3_4.tfrecord") as f:
    for example in [person3, person4]:
        f.write(example.SerializeToString())
        print('Saved: ',example.features.feature['id'])

# Load Examples from files
# 2 steps:
#   1. MUST create a feature description to parse
#   2. Load the data
feature_description = {
    "name": tf.io.FixedLenFeature([], tf.string, default_value=""), # MUST specify shape [], and type
    "id": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "emails": tf.io.VarLenFeature(tf.string), # ONLY requires shape
}
# Load examles separately
dataset = tf.data.TFRecordDataset(["datasets/person3_4.tfrecord"])
for serialized_example in dataset:
    parsed_example = tf.io.parse_single_example(serialized_example, feature_description)
    print('Loaded: ',parsed_example['id'])
# Load examples in batch
dataset = tf.data.TFRecordDataset(["datasets/person3_4.tfrecord"]).batch(5)
for serialized_example in dataset:
    parsed_examples = tf.io.parse_example(serialized_example, feature_description)
    print('Loaded batch: ',parsed_examples['id'])

# %% 21.3. SEQUENCEEXAMPLE protobuf
# INFO: To store features that are LISTS OF LISTS
import tensorflow as tf
from tensorflow.train import BytesList, FloatList, Int64List  
from tensorflow.train import Feature, Features, Example
from tensorflow.train import FeatureList, FeatureLists, SequenceExample
# Context features:
context_features = Features(feature={
    "author_id": Feature(int64_list=Int64List(value=[123])),
    "title": Feature(bytes_list=BytesList(value=[b"A", b"desert", b"place", b"."])),
    "pub_date": Feature(int64_list=Int64List(value=[1623, 12, 25]))
})
# Content feature:
content = [['CONTENT1:', "When", "shall", "we", "three", "meet", "again", "?"],
           ['CONTENT2:', "In", "thunder", ",", "lightning", ",", "or", "in", "rain", "?"]]
# Comment feature:
comments = [['COMMENT1:', "When", "the", "hurlyburly", "'s", "done", "."],
            ['COMMENT2:',"When", "the", "battle", "'s", "lost", "and", "won", "."]]

# Convert content and comment to LISTS OF LISTS (LISTS OF sentences, each sentence is a LIST of words):
def words_to_feature(words):
    return Feature(bytes_list=BytesList(value=[word.encode("utf-8") for word in words]))
content_features = [words_to_feature(sentence) for sentence in content]
comments_features = [words_to_feature(comment) for comment in comments]

# Make an example of Context, Content and Comment features:   
# NOTE: 'context', 'feature_lists' are COMPULSORY keywords in Sequence Example .  
example1 = SequenceExample(
    context=context_features,
    feature_lists=FeatureLists(feature_list={
        "content": FeatureList(feature=content_features),
        "comments": FeatureList(feature=comments_features)
    }))

print(example1.context.feature['pub_date'])
print(example1.feature_lists.feature_list['comments'])

# Save to file:
serialized_example1 = example1.SerializeToString()
with tf.io.TFRecordWriter('datasets/example1.tfrecord') as f:
    f.write(serialized_example1)
    
# Read Sequence Example from file:
context_feature_descriptions = {
    "author_id": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "title": tf.io.VarLenFeature(tf.string),
    "pub_date": tf.io.FixedLenFeature([3], tf.int64, default_value=[0, 0, 0]) }
list_feature_descriptions = {
    "content": tf.io.VarLenFeature(tf.string),
    "comments": tf.io.VarLenFeature(tf.string) }
dataset = tf.data.TFRecordDataset(["datasets/example1.tfrecord"]).batch(5)
for serialized_example in dataset:
    parsed_context, parsed_feature_lists = tf.io.parse_single_sequence_example(
        serialized_example1, context_feature_descriptions, list_feature_descriptions)
    print('Loaded author id: ',parsed_context['author_id'])
    #print('Loaded comments: ',parsed_feature_lists['comments'])
    #comments = tf.RaggedTensor.from_sparse(parsed_feature_lists["comments"])
    comments = tf.sparse.to_dense(parsed_feature_lists['comments']).numpy()
    print('\nLoaded batch: ',comments)


#%% 21.4. Store IMAGE with Example protobuf and TF record
# NOTE: NOT much faster than using raw images.
# Load images:
import cv2
img = cv2.imread('images/HCMUTE.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
import matplotlib.pyplot as plt
plt.imshow(img)
img2 = cv2.imread('images/CanhDong_Hue_ivivu.jpg')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
# Store images to tf file:
with tf.io.TFRecordWriter("images/imgs.tfrecord") as f:
    for img in [img, img2]:
        data = tf.io.encode_jpeg(img)
        img_example = Example(features=Features(feature={
            "image": Feature(bytes_list=BytesList(value=[data.numpy()]))}))
        f.write(img_example.SerializeToString())
# Read images from tf record file:
feature_description = { "image": tf.io.VarLenFeature(tf.string) }
dataset = tf.data.TFRecordDataset(["images/imgs.tfrecord"])
for serialized_example in dataset:
    img_example = tf.io.parse_single_example(serialized_example, feature_description)
    decoded_img = tf.io.decode_jpeg(img_example["image"].values[0])
    plt.imshow(decoded_img)
    plt.show()

#%% 21.5. Store tensors in TF record files
t = tf.constant([[0., 1.], [2., 3.], [4., 5.]])
s = tf.io.serialize_tensor(t) # ready to save to file
tf.io.parse_tensor(s, out_type=tf.float32) #parse tensor read from file 

#endregion


''' WEEK 08 '''

# In[3]: PREPROCESSING THE INPUT FEATURES
# NOTE: tf Dataset is not so convenient for data processing, 
#    => May use sklearn, pandas... if possible
#region
# Load demo data: Boston housing data
(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.boston_housing.load_data()
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)
print('\nLoad Boston housing data (label: price in kUSD). Training size:', X_train.shape)
print('\nRaw data:\n', X_train[:2])

#%% 3.1. A layer for standardizing data
# NOTE: options for standardizing data
#   1. Use sklearn StandardScaler (recommended)
#   2. Add a batch-norm layer after input layer
#   3. Add a std layer (using keras) as below
class myStandardization(keras.layers.Layer): 
    # NOTE: 
    #   + It's required to call compute_mean_std() before using this layer in your net.
    #   + data_part is used to to compute means and standar deviations, hence, should be representative of your data 
    def __init__(self, data_part):
        super(myStandardization, self).__init__()
        #if data_part is not None:
        self.compute_mean_std(data_part)
    def compute_mean_std(self, data_part): # your method
        self.means = np.mean(data_part, axis=0, keepdims=True) 
        self.stds = np.std(data_part, axis=0, keepdims=True) 
    def call(self, inputs): # this method computes output of this layer
        return (inputs - self.means) / (self.stds + keras.backend.epsilon())

# Use myStandardization layer in a NN
std_layer = myStandardization(X_train) # or CALL: std_layer.compute_mean_std(X_train)
model = keras.Sequential([
            keras.layers.Input(X_train.shape[-1]),
            std_layer ])
print('\nScaled data (by myStd layer):\n', model(X_train[0]))
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
print('\nScaled data (by Sklearn):\n', X_train_scaled[0])


#%% 3.2. EMBEDDING (convert categorical data to numerical ones)
# NOTE: Embedding is MUCH better than one-hot encoding.
# >>>> See slides
#region
# Synthesize data
if 0:
    import csv
    import random
    list_vi_tri = ['Sài Gòn', 'Vũng Tàu', 'Nha Trang', 'Hà Nội', 'An Giang', 'Tây Ninh']
    no_of_samples = 10000
    with open('datasets/my_housing_data.csv', 'w', newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(['Diện tích','Số phòng','Vị trí','Giá'])
        for i in range(no_of_samples):
            dien_tich = np.random.randint(20,100)
            so_phong = np.random.randint(1,4)
            vi_tri = random.sample(list_vi_tri,1)[0]
            if vi_tri in ('Sài Gòn', 'Hà Nội'):
                gia = np.random.randint(2500,8000)
            elif vi_tri in ('An Giang', 'Tây Ninh'):
                gia = np.random.randint(500,3000)
            else: 
                gia = np.random.randint(1500,5000)
            writer.writerow([dien_tich,so_phong,vi_tri,gia])
        
# 3.2.1. Load and prepare data
import pandas
my_data_raw = pandas.read_csv('datasets/my_housing_data.csv')
X_full = my_data_raw.iloc[:,:-1].values
y_full = my_data_raw.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train_full, X_test, y_train_full, y_test = train_test_split(X_full, y_full, test_size=0.2)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.2)
from sklearn.preprocessing import StandardScaler
# For single num col:
# scaler = StandardScaler().fit(X_train[:,:2].reshape(-1,1))
# X_train[:,:2] = scaler.transform(X_train[:,:2].reshape(-1, 1)).T
# X_valid[:,:2] = scaler.transform(X_valid[:,:2].reshape(-1, 1)).T
# X_test[:,:2] = scaler.transform(X_test[:,:2].reshape(-1, 1)).T
# For >1 num cols:
scaler = StandardScaler().fit(X_train[:,:2])
X_train[:,:2] = scaler.transform(X_train[:,:2])
X_valid[:,:2] = scaler.transform(X_valid[:,:2])
X_test[:,:2] = scaler.transform(X_test[:,:2])

# Just for testing the td.dataset:
# batch_size = 32
# X_train_tf = tf.data.Dataset.from_tensor_slices(np.array(X_train[:,0],dtype=np.float32))
# X_train_tf = X_train_tf.batch(batch_size).prefetch(1)
# for i in X_train_tf.take(5):
#     print(i.numpy())

#%% 3.2.2. Convert categorical feature (Vị trí) to integers  
# NOTE: Each category will be convert to an embedding vector (of some dim M).
#       => hence, N categories <=> N embedding vectors.
#   In case you vocab does NOT contain all categories (eg., when new data come), 
#   => you have oov (out of vocabulary) categories.
#   => num_oov_buckets: max no. of additional embed vectors for new oov cats.
vocab = np.unique(X_train[:,-1]) # = ['An Giang', 'Hà Nội', 'Nha Trang',...]
print('\nVocab: ', vocab)
num_oov_buckets = 3

# Create a lookup table (to covert categories to integers)
table_init = tf.lookup.KeyValueTensorInitializer(
    keys=tf.constant(vocab), values=tf.range(len(vocab), dtype=tf.int64))
table = tf.lookup.StaticVocabularyTable(table_init, num_oov_buckets)
print('\nExamples of categories converted to id numbers:')
sample_cats = tf.constant(['An Giang', 'Sài Gòn', 'An Giang','new1','new2','new3'])
print(table.lookup(sample_cats))

# 3.2.2. Create a NN
# Create an Embedding layer (embeddings in this will be learned using training data)
embed_vec_dim = 2
embed_layer = keras.layers.Embedding(
    input_dim=len(vocab) + num_oov_buckets, # imagine the input is an one-hot vec 
    output_dim=embed_vec_dim, name='embed_layer')
print('\nPrint initial embedding vectors:')
init_embeds = []
for i in range(len(vocab)+num_oov_buckets):
    init_embeds.append(embed_layer(i).numpy())
    if i<len(vocab):
        print('Init embed of',vocab[i],':',init_embeds[i])
    else:
        print('Init embed',i,'(oov):',init_embeds[i])
# Plot the init embeddings
init_embeds = np.array(init_embeds)
plt.plot(init_embeds[:len(vocab),0],init_embeds[:len(vocab),1], marker='.', color='r', markersize=15, linestyle='None')
for i in range(len(vocab)):
    plt.text(init_embeds[i,0]+.002,init_embeds[i,1]+.002,vocab[i],fontsize=11)
plt.title('Initial embeddings of Vị trí')
plt.axis([-.05, .05, -.05, .07])
plt.show()

#%% Create a NN with embedding layers
num_inputs = keras.layers.Input(shape=[2], dtype=tf.float32)
cat_input = keras.layers.Input(shape=[], dtype=tf.string)
cat_indices = keras.layers.Lambda(lambda cats: table.lookup(cats))(cat_input)
cat_embed = embed_layer(cat_indices)
concaten = keras.layers.concatenate([num_inputs, cat_embed])
hidden1 = keras.layers.Dense(10,activation='elu',kernel_initializer='he_normal')(concaten)
hidden2 = keras.layers.Dense(10,activation='elu',kernel_initializer='he_normal')(hidden1)
hidden3 = keras.layers.Dense(10,activation='elu',kernel_initializer='he_normal')(hidden2)
output = keras.layers.Dense(1)(hidden3)
model = keras.models.Model(inputs=[num_inputs, cat_input], outputs=[output])
model.save('models/my_housing_fresh.h5')
model.summary()
keras.utils.plot_model(model, "models/housing_model.png", show_shapes=True)

#%% 3.2.3. Train the NN 
# Find a good learning rate
def qFindLearningRate(model, X_train, y_train, increase_factor = 1.005, batch_size=32, fig_name='find_lr'):
    # Create a callback to increase the learning rate after each batch, store losses to plot later
    class IncreaseLearningRate_cb(keras.callbacks.Callback):
        def __init__(self, factor):
            self.factor = factor
            self.rates = []
            self.losses = []
        def on_batch_end(self, batch, logs):
            K = keras.backend
            self.rates.append(K.get_value(self.model.optimizer.lr))
            self.losses.append(logs["loss"])
            K.set_value(self.model.optimizer.lr, self.model.optimizer.lr * self.factor)
    increase_lr = IncreaseLearningRate_cb(factor=increase_factor)

    # Train 1 epoch
    history = model.fit(X_train, y_train, epochs=1, batch_size=batch_size, callbacks=[increase_lr])

    # Plot losses after training batches. 
    # NOTE: a batch has a different learning rate 
    from statistics import median
    plt.plot(increase_lr.rates, increase_lr.losses)
    plt.gca().set_xscale('log')
    #plt.hlines(min(increase_lr.losses), min(increase_lr.rates), max(increase_lr.rates))
    plt.axis([min(increase_lr.rates), max(increase_lr.rates), min(increase_lr.losses), median(increase_lr.losses)])
    plt.grid()
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")
    plt.savefig(fig_name, dpi=300)
    plt.show()
    return increase_lr
batch_size = 128 # NOTE: large batch_size (>32) ONLY for with 1cycle lr scheduling
if 0: 
    model = keras.models.load_model('models/my_housing_fresh.h5')
    init_lr = 1e-5
    #model.compile(optimizer=keras.optimizers.SGD(lr=init_lr,momentum=0.9,nesterov=True), loss='mae', metrics='accuracy')
    model.compile(optimizer=keras.optimizers.Nadam(lr=init_lr), loss='mae', metrics='accuracy')
    increase_lr = qFindLearningRate(model, (X_train[:,:-1].astype('float32'),X_train[:,-1]), y_train, fig_name='find_lr', batch_size=batch_size)
# => Good learning rate: 
#good_lr = 1e-2

#%% Train with 1Cycle lr scheduling
# NOTE: Very FAST and HIGH ACCURACY. A MUST try!!
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
data_train = (X_train[:,:-1].astype('float32'),X_train[:,-1])
data_valid = (X_valid[:,:-1].astype('float32'),X_valid[:,-1])

new_training = 0
if new_training:
    #batch_size = 128 
    good_lr = 0.02

    n_epochs = 20
    init_lr = good_lr/10
    model = keras.models.load_model('models/my_housing_fresh.h5')
    model.compile(optimizer=keras.optimizers.Nadam(lr=init_lr), loss='mae') # metrics='mae'
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=n_epochs/2)
    save_checkpoint = keras.callbacks.ModelCheckpoint('models/my_housing_best_1cycle.h5', save_best_only=True)
    #log_dir = "logs/CIFAR10_qCUSTOM_1cycle_" + datetime.datetime.now().strftime("%m%d-%H%M")
    #tensor_board = keras.callbacks.TensorBoard(log_dir=log_dir)

    n_iters = int(len(X_train) / batch_size) * n_epochs
    onecycle = OneCycleScheduler(n_iters, max_rate=good_lr)

    #with tf.device('/gpu:0'): # to use GPU (default of Keras is GPU if any)    
    history = model.fit(data_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_data=(data_valid,y_valid), 
        callbacks=[early_stop, save_checkpoint, onecycle])
    history = history.history
model = keras.models.load_model('models/my_housing_best_1cycle.h5')
model.evaluate(data_valid, y_valid)

#%% Train with performance sched
new_training = 0
if new_training:
    model = keras.models.load_model('models/my_housing_fresh.h5')
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=6)
    performance_sched = keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=2)
    model_save = keras.callbacks.ModelCheckpoint('models/my_housing_best_perforSched.h5',save_best_only=True)

    #model.compile(optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9),loss='mae')
    model.compile(optimizer=keras.optimizers.Nadam(lr=0.02),loss='mae')
    model.fit((X_train[:,:-1].astype('float32'),X_train[:,-1]), y_train, epochs=20, batch_size=batch_size, validation_data = ((X_valid[:,:-1].astype('float32'),X_valid[:,-1]), y_valid),
          callbacks=[early_stop, performance_sched, model_save])
model = keras.models.load_model('models/my_housing_best_perforSched.h5')
model.evaluate(data_valid, y_valid)

#%% 3.2.4. See the embeddings
embed_layer = model.get_layer('embed_layer')  
embed_vecs = embed_layer.get_weights()
embed_vecs = np.array(embed_vecs[0])
init_embeds = np.array(init_embeds)

plt.plot(init_embeds[:len(vocab),0],init_embeds[:len(vocab),1], marker='.', color='r', markersize=15, linestyle='None')
plt.plot(embed_vecs[:len(vocab),0],embed_vecs[:len(vocab),1], marker='s', color='b', markersize=8, linestyle='None')
for i in range(len(vocab)):
    plt.text(init_embeds[i,0]+.02,init_embeds[i,1]+.02,vocab[i],fontsize=7)
    plt.text(embed_vecs[i,0]+.02,embed_vecs[i,1]+.02,vocab[i],fontsize=7)
plt.text(.3,.48,'Tỉnh',fontsize=15,alpha=.4,fontstyle='italic')
plt.text(.8,1.05,'TP biển',fontsize=15,alpha=.4,fontstyle='italic')
plt.text(1.55,1.48,'TP lớn',fontsize=15,alpha=.4,fontstyle='italic')
plt.legend(['Init embed','Learned embed'])
plt.title('Embeddings of Vị trí')
plt.show()
#endregion

#endregion


# In[4]: TF DATASETS
#region
# NOTE: Use tfds dataset: https://www.tensorflow.org/datasets/overview
#   Iterate over a dataset:
#       As dict (default)
#       As tuple: as_supervised=True
#       As numpy: tfds.as_numpy: convert tf.Tensor -> np.array

# 4.1. Load data
import tensorflow_datasets as tfds 
dataset, info = tfds.load(
    name='mnist', # name of dataset: https://www.tensorflow.org/datasets/catalog/overview
    with_info=True, # get info of the data
    #split=['train', 'test'], # split: parts of data to load. NOTE: using this turn the dataset to a LIST, not a DICT anymore, hence dataset[0]=train_set, dataset[1]=test_set. See https://www.tensorflow.org/datasets/splits
    #batch_size=-1, # load the full dataset in a single batch
    #as_supervised=True, # load data as Tuple, instead of Tensors.
    )
mnist_train, mnist_test = dataset["train"], dataset["test"]
train_size = info.splits['train'].num_examples
test_size = info.splits['test'].num_examples

# 4.2. Convert dictionay to tuples
# NOTE: Each item in mnist_train is a DICTIONARY of features (image pixels) and label.
#       But keras requires training data as TUPLES of (features, label)
#       => Use map() to convert.
print('\nBefore converting (dictionary):')
for item in mnist_train.take(3): 
    #print(item["image"])
    print(item["label"])
mnist_train = mnist_train.map(lambda item: (item["image"], item["label"]))    
print('\nAfter converting (tuples):')
for item in mnist_train.take(3): 
    #print('Features:',item[0])
    print('Label:',item[1])

# 4.3. Preprocess data (shuffle, prefetch...)
mnist_train = mnist_train.shuffle(10000).batch(32).prefetch(1)
print('\nAfter processing:')
for item in mnist_train.take(3): 
    #print('Features:',item[0])
    print('Labels:',item[1])

# 4.4. (demo) Train a NN
import tensorflow.keras.layers as kl
model = tf.keras.models.Sequential([
    kl.Input(shape=(28, 28)),
    kl.Flatten(),
    kl.Dense(20, activation='elu', kernel_initializer='he_normal'),
    kl.Dense(10) ])
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam")
model.fit(mnist_train, epochs=1)

#endregion
 