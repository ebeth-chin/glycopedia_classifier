# 03_finetune.py
# purpose: fine tune inception using Food 101
# adapted for atlas
# E. Chin 09/21/2021

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

#set up GPUs on Atlas
os.environ['TF_GPU_THREAD_MODE']='gpu_private'
tf.random.set_seed(1234)
np.random.seed(1234)
K.clear_session()

#check GPUs
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
# Make the model w/ new classification head
####
with strategy.scope():
#feature extraction
    base_model = InceptionV3(weights='imagenet', include_top=False)
    base_model.trainable = False
    inputs = keras.Input(shape = (299,299,3))
    x = base_model(inputs, training = False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    outputs = keras.layers.Dense(n_classes, activation = "softmax")(x)
    model = Model(inputs=inputs,outputs= outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate), loss="categorical_crossentropy",metrics = ['accuracy'])

model.summary()

model_path = 'models/'+ datetime.now().strftime("%Y%m%d-%H%M%S")


hist = model.fit(
    ds_tf,
    validation_data=val_tf,
    epochs=5,
    use_multiprocessing = True,
    workers = 48,
    verbose = 1)

#plot
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


#save the model w/ new classification head
task_type, task_id = (strategy.cluster_resolver.task_type,
                      strategy.cluster_resolver.task_id)
write_path = write_filepath(model_path, task_type, task_id)

model.save(write_path)

#evaluate the model w/ new classification head
print('Evaluating test data for new model...')
test_loss, test_accuracy = model.evaluate(test_tf)
print('Test accuracy for new model:', test_accuracy)
print('Test loss for new model:', test_loss)


#remove the temporary/non-chief worker models:
if not _is_chief(task_type, task_id):
  tf.io.gfile.rmtree(os.path.dirname(write_path))

######
#fine tune the last 250 layers of inception
#####
with strategy.scope():
    #fine tune
    base_model.trainable = True
    for layer in base_model.layers[:249]: #fine tune the last 2 inception blocks (last 250 layers)
        layer.trainable = False
    for layer in base_model.layers[249:]:
        layer.trainable = True

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate/10), loss="categorical_crossentropy", metrics = ['accuracy'])

model.summary()

ckpt_backups = 'ckpt_backups/' #path to where you want to save the checkpoints
callbacks = [tf.keras.callbacks.experimental.BackupAndRestore(backup_dir=ckpt_backups)]

finetune_hist= model.fit(
    ds_tf,
    validation_data=val_tf,
    epochs=10,
    use_multiprocessing = True,
    workers = 48,
    verbose = 1,
    callbacks = callbacks)

#save the fine tuned model
task_type, task_id = (strategy.cluster_resolver.task_type,
                      strategy.cluster_resolver.task_id)
finetuned_write_path = write_filepath(model_path, task_type, task_id)

model.save(finetuned_write_path)

#plot
plt.plot(finetune_hist.history['accuracy'], label='Training Acc')
plt.plot(finetune_hist.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc="upper left")
plt.savefig('accuracy_finetuned.jpg')

plt.plot(finetune_hist.history['loss'], label='Training Loss)')
plt.plot(finetune_hist.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (Categorical Crossentropy)')
plt.legend(loc="upper right")
plt.savefig('loss_finetuned.jpg')


#remove work from non-chief workers
if not _is_chief(task_type, task_id):
  tf.io.gfile.rmtree(os.path.dirname(finetuned_write_path))

print('Total Train Time:', time()-start)

#evaluate the test data on the fine tuned model
print('Evaluating test data for finetuned model...')
test_loss, test_accuracy = model.evaluate(test_tf)
print('Test accuracy for finetuned model:', test_accuracy)
print('Test loss for finetuned model:', test_loss)
