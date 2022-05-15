
""" Import Libraries"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import cv2
from os import listdir
import glob
import re
import math 
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import random
import math
from skimage import exposure
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, BatchNormalization, Dropout, Flatten, Dense, Activation, MaxPool2D, Conv2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.utils import to_categorical
from keras import optimizers
from tensorflow.keras.optimizers import Adam
from keras.callbacks import Callback,ModelCheckpoint,ReduceLROnPlateau
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
import keras.backend as K
tf.config.run_functions_eagerly(True)


"""### Configs"""

configs = {
    "seed": 112,
    "epochs": 15,
    "batch_size": 16,
    "num_classes": 10,
    "image_size": 224,
}


#Functions
auto = tf.data.AUTOTUNE

def make_datasets(images, labels, is_train=False):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    batch_size=configs['batch_size']
    if is_train:
        dataset = dataset.shuffle(batch_size * 10)
    dataset = dataset.batch(batch_size)
    
    return dataset.prefetch(auto)

def f1_score(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


# Dataset Preparation

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train.astype("float32") #/ 255
x_test = x_test.astype("float32") #/ 255

train_dataset = make_datasets(x_train, y_train, is_train=True)
val_dataset = make_datasets(x_test, y_test)


#LOAD PRE-TRAINED model: ResNet50, remove fully connected layers and freeze other layers
base_model = tf.keras.applications.ResNet50(input_shape=(configs['image_size'],configs['image_size'],3),include_top=False,weights="imagenet")#ResNet152V2#InceptionResNetV2

# Freezing Layers
for layer in base_model.layers[:-10]:
    layer.trainable=False


# Building Transfer Learning Model
model=Sequential()
model.add(base_model)
model.add(Dropout(0.5))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(256,kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(128,kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(32,kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(configs['num_classes'],activation='softmax'))

# Model Summary
model.summary()

#Add Model's Evaluation Metrics
METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),  
      tf.keras.metrics.AUC(name='auc'),
        f1_score,
]

#Define Callbacks
lrd = ReduceLROnPlateau(monitor = 'val_loss',patience = 5,verbose = 1,factor = 0.8, min_lr = 1e-6)

#Track Model Checkpoint
mcp = ModelCheckpoint('V2_TF_ResNet50.h5')
es = EarlyStopping(verbose=1, patience=5)

#Compile model
model.compile(optimizer='Adam', loss="categorical_crossentropy", metrics=METRICS)



#Model Training
history = model.fit(
        train_dataset,
        validation_data=val_dataset,epochs=configs['epochs'],callbacks=[lrd,mcp,es])

#Calculate Average Model's Training Validation Accuracy
avg=0
for value in history.history['val_accuracy']:
  avg=avg+value

print(avg/len(history.history['val_accuracy']))


#save model
model.save('TF_ResNet_50.h5')

# Summarize & visualize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for precision
plt.plot(history.history['precision'])
plt.plot(history.history['val_precision'])
plt.title('model precision')
plt.ylabel('precision')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for f1_score
plt.plot(history.history['f1_score'])
plt.plot(history.history['val_f1_score'])
plt.title('model f1_score')
plt.ylabel('f1_score')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Save history to csv
pd.DataFrame(history.history).to_csv("resnet50_history.csv")

