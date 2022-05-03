import numpy as np
import pandas as pd
from tensorflow import keras
from keras import layers
from keras.models import load_model
from keras.applications.resnet import ResNet50
from keras.optimizers import adam_v2
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing import image


def modelBuilding():
    # Importing Convolutional Base

    conv_base = ResNet50(weights='imagenet',
                         include_top=False,
                         input_shape=(180, 180, 3))
    conv_base.trainable = False

    # Fully Connected Neural Network Head
    cnn_model = keras.Sequential([
        conv_base,
        layers.Flatten(),
        layers.Dense(300, activation='relu'),
        layers.Dense(1, activation='sigmoid')])

    # Compilation with loss function, optimizers and eval metrics
    cnn_model.compile(loss='binary_crossentropy',
                      optimizer=adam_v2.Adam(learning_rate=2e-5),
                      metrics=['accuracy'])
    print("Compilation was successful...")
    return cnn_model


def modelFitting(train_gen, val_gen):
    cnn = modelBuilding()
    print(cnn.summary())

    # Creating Checkpoints
    checkpoint_cb = ModelCheckpoint("face_mask.h5",
                                    save_best_only=True)
    early_stopping_cb = EarlyStopping(min_delta=0.001,
                                      patience=4,
                                      restore_best_weights=True)
    print("Checkpoints successfully created...")

    try:
        history = cnn.fit(train_gen,
                          steps_per_epoch=40,
                          epochs=40,
                          validation_data=val_gen,
                          validation_steps=30,
                          callbacks=[checkpoint_cb, early_stopping_cb],
                          verbose=1)
        print("model has finished training...")

        return history
    except Exception as e:
        print(e)
