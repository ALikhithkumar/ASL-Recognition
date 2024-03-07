# -*- coding: utf-8 -*-
"""ASL Implementation 99 Accuracy.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_4FX0JnLl3qTynNNJFRyuaztO9FYORKb

### 1. Importing packages <a id=1></a>
"""

# Commented out IPython magic to ensure Python compatibility.
# import data processing and visualisation libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

# import image processing libraries
import cv2
import skimage
from skimage.transform import resize

# import tensorflow and keras
import tensorflow as tf
from tensorflow import keras
import os

print("Packages imported...")

# Commented out IPython magic to ensure Python compatibility.
# %%writefile kaggle.json
# {"username":"sairamparshi07","key":"a07ce4e9f16b95b9b0c358f5b12deec7"}

import os

def download_kaggle_dataset(dataset_path):
    # Kaggle dataset path example: 'grassknoted/asl-alphabet'
    # Ensure Kaggle API is installed
    os.system('pip install -q kaggle')

    # Check for kaggle.json file
    if not os.path.isfile('kaggle.json'):
        print("kaggle.json file not found. Please upload it.")
        return

    # Set up Kaggle directory and permissions
    os.system('mkdir -p ~/.kaggle')
    os.system('cp kaggle.json ~/.kaggle/')
    os.system('chmod 600 ~/.kaggle/kaggle.json')

    # Download the dataset
    os.system(f'kaggle datasets download -d {dataset_path}')

    # Unzip the dataset
    zip_file = f'{dataset_path.split("/")[-1]}.zip'
    os.system(f'unzip -q {zip_file}')

# Example Usage:
download_kaggle_dataset('grassknoted/asl-alphabet')

"""### 2. Importing the dataset from training directory <a id=2></a>"""

batch_size = 64
imageSize = 64
target_dims = (imageSize, imageSize, 3)
num_classes = 29

train_len = 87000
train_dir = '/content/asl_alphabet_train/asl_alphabet_train/'

def get_data(folder):
    X = np.empty((train_len, imageSize, imageSize, 3), dtype=np.float32)
    y = np.empty((train_len,), dtype=np.int)
    cnt = 0
    for folderName in os.listdir(folder):
        if not folderName.startswith('.'):
            if folderName in ['A']:
                label = 0
            elif folderName in ['B']:
                label = 1
            elif folderName in ['C']:
                label = 2
            elif folderName in ['D']:
                label = 3
            elif folderName in ['E']:
                label = 4
            elif folderName in ['F']:
                label = 5
            elif folderName in ['G']:
                label = 6
            elif folderName in ['H']:
                label = 7
            elif folderName in ['I']:
                label = 8
            elif folderName in ['J']:
                label = 9
            elif folderName in ['K']:
                label = 10
            elif folderName in ['L']:
                label = 11
            elif folderName in ['M']:
                label = 12
            elif folderName in ['N']:
                label = 13
            elif folderName in ['O']:
                label = 14
            elif folderName in ['P']:
                label = 15
            elif folderName in ['Q']:
                label = 16
            elif folderName in ['R']:
                label = 17
            elif folderName in ['S']:
                label = 18
            elif folderName in ['T']:
                label = 19
            elif folderName in ['U']:
                label = 20
            elif folderName in ['V']:
                label = 21
            elif folderName in ['W']:
                label = 22
            elif folderName in ['X']:
                label = 23
            elif folderName in ['Y']:
                label = 24
            elif folderName in ['Z']:
                label = 25
            elif folderName in ['del']:
                label = 26
            elif folderName in ['nothing']:
                label = 27
            elif folderName in ['space']:
                label = 28
            else:
                label = 29
            for image_filename in os.listdir(folder + folderName):
                img_file = cv2.imread(folder + folderName + '/' + image_filename)
                if img_file is not None:
                    img_file = skimage.transform.resize(img_file, (imageSize, imageSize, 3))
                    img_arr = np.asarray(img_file).reshape((-1, imageSize, imageSize, 3))

                    X[cnt] = img_arr
                    y[cnt] = label
                    cnt += 1
    return X,y
X_train, y_train = get_data(train_dir)
print("Images successfully imported...")

X_train

"""#### 2.1 Checking the shape of data <a id=3></a>"""

print("The shape of X_train is : ", X_train.shape)
print("The shape of y_train is : ", y_train.shape)

"""#### 2.2 Checking the shape of one image <a id=4></a>"""

print("The shape of one image is : ", X_train[0].shape)

"""#### 2.3 Viewing the image <a id=5></a>"""

plt.imshow(X_train[0])
plt.show()

"""##### 2.3.1 Making copies of original data"""

X_data = X_train
y_data = y_train
print("Copies made...")

"""### 3. Data processing <a id=6></a>

#### 3.1 Train/test split <a id=7></a>
"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3,random_state=42,stratify=y_data)

"""#### 3.2 One-Hot-Encoding <a id=8></a>"""

# One-Hot-Encoding the categorical data
from tensorflow.keras.utils import to_categorical
y_cat_train = to_categorical(y_train,29)
y_cat_test = to_categorical(y_test,29)

"""#### 3.3 Dimension Check of variables <a id=9></a>"""

# Checking the dimensions of all the variables
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print(y_cat_train.shape)
print(y_cat_test.shape)

"""### 4. Garbage Collection <a id=10></a>"""

# This is done to save CPU and RAM space while working on Kaggle Kernels. This will delete the specified data and save some space!
import gc
del X_data
del y_data
gc.collect()

"""### 5. Modeling <a id=11></a>

#### 5.1 Importing packages <a id=12></a>
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
print("Packages imported...")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall

# Enable Mixed Precision Training
if tf.config.list_physical_devices('GPU'):
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print('Mixed precision enabled')

# Confirming that GPU is available
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if not gpu_devices:
    print('No GPU found. Using CPU')
else:
    print(f'GPUs found: {gpu_devices}')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)

"""#### 5.2 Building model <a id=13></a>"""

model = Sequential()

model.add(Conv2D(32, (5, 5), input_shape=(64, 64, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(29, activation='softmax'))

model.summary()

"""#### 5.3 Early Stopping and Compiling <a id=14></a>

##### 5.3.1 Early Stopping

Early Stopping is done to make sure the model fitting stops at the most optimized accuracy point. After the early stopping point, the model might start overfitting. For testing purposes, this step can be skipped and complete training can be done.
"""

# Early Stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=2)

"""##### 5.3.2 Compiling"""

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

"""#### 5.4 Model fitting <a id=15></a>"""

# Train the model
model.fit(X_train, y_cat_train, epochs=50, batch_size=64, verbose=2, validation_data=(X_test, y_cat_test), callbacks=[early_stop])

"""#### 5.5 Model metrics <a id=16></a>

##### 5.5.1 Metrics from model history
"""

metrics = pd.DataFrame(model.history.history)
print("The model metrics are")
metrics

"""##### 5.5.2 Plotting the training loss"""

metrics[['loss','val_loss']].plot()
plt.show()

"""##### 5.5.3 Plotting the testing loss"""

metrics[['accuracy','val_accuracy']].plot()
plt.show()

"""##### 5.5.4 Model evaluation"""

model.evaluate(X_test,y_cat_test,verbose=0)

"""#### 5.6 Predictions <a id=17></a>"""

# Define batch size for prediction
prediction_batch_size = 32  # Adjust this based on your RAM availability

# Initialize an empty array to store predictions
predictions = []

# Predict in batches
for i in range(0, len(X_test), prediction_batch_size):
    batch = X_test[i:i+prediction_batch_size]
    batch_probabilities = model.predict(batch)
    batch_predictions = np.argmax(batch_probabilities, axis=1)
    predictions.extend(batch_predictions)

# Convert predictions list to a numpy array
predictions = np.array(predictions)

print("Predictions done...")

"""##### 5.6.1 Classification report"""

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,predictions))

"""##### 5.6.2 Confusion matrix heatmap"""

plt.figure(figsize=(12,12))
sns.heatmap(confusion_matrix(y_test,predictions))
plt.show()

"""#### 5.7 Saving the model <a id=18></a>"""

# from keras.models import load_model
model.save('ASL.h5')
print("Model saved successfully...")

"""[back to top](#19)"""
