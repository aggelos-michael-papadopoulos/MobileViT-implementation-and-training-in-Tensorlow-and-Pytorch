from tensorflow.keras import losses
import keras.callbacks
from mobile_vit_tensorflow import create_mobilevit
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import tensorflow as tf
import numpy as np
import random
import os
import cv2
import tqdm
import time
import wandb
from wandb.keras import WandbCallback
import tensorflow_addons as tfa
# anestis
from anestis import efficientformer_l1

# run with lower GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


#  MODELS
def mobilevit_xxs():
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    return create_mobilevit(channels, dims, num_classes=257, expansion_factor=2)


def mobilevit_xs():
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    return create_mobilevit(channels, dims, num_classes=257)


def mobilevit_s():
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    return create_mobilevit(channels, dims, num_classes=257)


WEIGHTS_PATH = '/home/angepapa/PycharmProjects/Mobile-vit/weights/tensorflow_weights/'

# config file
CFG = {'name': 'Mobile-ViT',
       'implementation': 'Tensorflow',
       'image_size': 256,
       'epochs': 300,
       'batch_size': 20,
       'initial_lr': 0.002,
       'weight_decay': 0.01,
       'seed': 42,
       'scheduler': 'None',
       'wandb': True}

# show CFG
for i in CFG:
    print(f'{i}: {CFG[i]}')

# Activate weights and biases
if CFG['wandb']:
    wandb.init(project='mobile-vit', entity='angepapa', config=CFG)


# Set randomSeed for reproducability
def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    np.random.seed(seed)
    tf.random.set_seed(seed)


seed_torch(seed=CFG['seed'])

# if __name__ == '__main__':
# Dataset creation for Dataloader
train_dir = '/home/angepapa/PycharmProjects/Mobile-vit/calteck_256/train'
val_dir = '/home/angepapa/PycharmProjects/Mobile-vit/calteck_256/val'

train_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True, zoom_range=0.2)
val_datagen = ImageDataGenerator(rescale=1. / 255)

train_loader = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(CFG['image_size'], CFG['image_size']),
    batch_size=CFG['batch_size'],  # put classes=[001.fk, ....] mode
    class_mode='categorical'
)

val_loader = val_datagen.flow_from_directory(
    directory=val_dir,
    target_size=(CFG['image_size'], CFG['image_size']),
    batch_size=CFG['batch_size'],  # put classes=[001.fk, ....] mode
    class_mode='categorical'
)

# Create model
model = mobilevit_xxs()
model.summary()

loss_fn = losses.CategoricalCrossentropy(label_smoothing=0.1)

model.compile(loss=loss_fn,
              optimizer='adam',
              metrics=['accuracy'])

# PLACE CALLBACKS HERE
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8,
                                                 patience=5, min_lr=1e-5, verbose=1)


model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=WEIGHTS_PATH,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

wandb_callbacks = WandbCallback()

callbacks = [wandb_callbacks, reduce_lr, model_checkpoint_callback]


model.fit_generator(
    generator=train_loader,
    steps_per_epoch=len(train_loader) // CFG['batch_size'],
    epochs=CFG['epochs'],
    validation_data=val_loader,
    validation_steps=len(val_loader) // CFG['batch_size'],
    callbacks=callbacks)

# model.save_weights('first_try.h5')
