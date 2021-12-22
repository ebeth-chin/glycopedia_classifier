# 02_tiny_cnn.py
# purpose: fitting a small cnn to make sure general workflow is ok
# adapted for atlas
# E. Chin 09/16/2021

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers, models, layers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow import keras
import numpy as np
import os
import json
import socket
import random
from time import time
from datetime import datetime
from glob import glob
import matplotlib.pyplot as plt
import pathlib
from tf_helper import _is_chief, _get_temp_dir, write_filepath

from functools import partial

start = time()

#set up GPUs to work on Atlas
os.environ['TF_GPU_THREAD_MODE']='gpu_private'
tf.random.set_seed(1234)
np.random.seed(1234)
K.clear_session()

print(tf.__version__)
print(tf.test.gpu_device_name())
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)

tf_config = json.loads(os.environ['TF_CONFIG'])

hostname = socket.gethostname()
inx=0

shortname = hostname.split('.')[0]

for name in tf_config['cluster']['worker']:
  currHost = name.split(':')[0]
  if shortname != currHost:
    inx+=1
  else:
    break;

tf_config['task']['index'] = inx
os.environ['TF_CONFIG'] = json.dumps(tf_config)

#set strategy
strategy = tf.distribute.MultiWorkerMirroredStrategy()
print('Number of devices in sync: %d' % strategy.num_replicas_in_sync)

batch_size_per_replica = 16
BATCH_SIZE = batch_size_per_replica * strategy.num_replicas_in_sync

####
# Import data
####

img_size = 299
SEED=1234
base_learning_rate = 3e-4
AUTOTUNE = tf.data.AUTOTUNE


#change this to the specific dataset you want to use
train_data_dir = '/food-101/data_3class/train'
validation_data_dir = '/food-101/data_3class/val'
test_data_dir = '/food-101/data_3class/test'

n_classes = len(os.listdir(train_data_dir))
print('number of classes:', n_classes)

train_data = tf.keras.preprocessing.image_dataset_from_directory(train_data_dir, labels = "inferred", color_mode = "rgb", image_size = (299,299),
                                                                 label_mode = "categorical", batch_size = BATCH_SIZE)
val_data = tf.keras.preprocessing.image_dataset_from_directory(validation_data_dir, labels = "inferred",
                                                               color_mode = "rgb", image_size = (299,299), label_mode = "categorical", batch_size = BATCH_SIZE)
test_data = tf.keras.preprocessing.image_dataset_from_directory(test_data_dir, labels = "inferred",
                                                               color_mode = "rgb", image_size = (299,299), label_mode = "categorical", batch_size = BATCH_SIZE)

####
# Augmentations
####
resize_and_rescale = tf.keras.Sequential([
  layers.experimental.preprocessing.Resizing(img_size, img_size),
  layers.experimental.preprocessing.Rescaling(1./255)
])

def process_image(image, label, img_size, augment = False):
    if augment:
        # apply simple augmentations
        image = tf.image.random_flip_left_right(image, seed = SEED)
        image = tf.image.random_brightness(image, max_delta = 0.6, seed = SEED)
        image = tf.image.random_contrast(image, lower = 0, upper = 0.9, seed = SEED)

    image= resize_and_rescale(image)

    return image, label

ds_tf = train_data.map(partial(process_image, img_size=299, augment = True),num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
val_tf = val_data.map(partial(process_image, img_size=299, augment = False),num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
test_tf = test_data.map(partial(process_image, img_size=299, augment = False),num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

####
# Make the model
####

def create_model(input_shape): #we have to specify the shape here
    return models.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu', kernel_regularizer = tf.keras.regularizers.l2(l = 0.001)),
        layers.MaxPooling2D(),
        layers.Dropout(rate = 0.5, seed = SEED),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer = tf.keras.regularizers.l2(l=0.01)),
        layers.Dropout(rate = 0.5, seed = SEED),
        layers.Dense(n_classes, activation='softmax')])

with strategy.scope():
    model = create_model((299,299,3))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate), loss="categorical_crossentropy",metrics = ['accuracy'])

model.summary()

model_path = 'models/'+ datetime.now().strftime("%Y%m%d-%H%M%S")

hist = model.fit(
    ds_tf,
    validation_data=val_tf,
    epochs=20,
    use_multiprocessing = True,
    workers = 48,
    verbose = 1)


task_type, task_id = (strategy.cluster_resolver.task_type,
                      strategy.cluster_resolver.task_id)
write_path = write_filepath(model_path, task_type, task_id)

model.save(model_path)

print('Total Train Time:', time()-start)

print('Evaluating test data...')


test_loss, test_accuracy = model.evaluate(test_tf)
print('Test accuracy:', test_accuracy)
print('Test loss:', test_loss)


#you can plot the training if you want:
plt.plot(hist.history['accuracy'], label='Training Acc')
plt.plot(hist.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc="upper left")
plt.savefig('accuracy.jpg')

plt.plot(hist.history['loss'], label='Training Loss)')
plt.plot(hist.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (Categorical Crossentropy)')
plt.legend(loc="upper right")
plt.savefig('loss.jpg')
