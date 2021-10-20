import numpy as np
import tensorflow as tf
from keras_preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from PIL import Image
import xml.etree.ElementTree as ET
from keras.applications.densenet import DenseNet121, preprocess_input
from keras.utils import np_utils
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.utils import *
from keras.callbacks import *
from tensorflow import keras
import pickle

TRAINING_DIR = os.path.join(os.getcwd(), "archive" ,"Training_path")
VALIDATION_DIR = os.path.join(os.getcwd(), "archive","Testing_path")


def ImageGenerators(TRAINING_DIR, VALIDATION_DIR):

    training_datagen = ImageDataGenerator(
        rescale = 1./255.,
        rotation_range = 40,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        fill_mode='nearest'
    )

    # Note that the validation data should not be augmented!
    test_datagen = ImageDataGenerator(
        rescale = 1./255.
    )

    train_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(224,224),
        class_mode='categorical',
        batch_size=32
    )

    validation_generator = test_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(224,224),
        class_mode='categorical',
        batch_size=32
    )

    label_map = (train_generator.class_indices)
    
    with open('label_classes.pkl', 'wb') as pick:
        pickle.dump(label_map, pick)

    return train_generator, validation_generator



def keras_model(train_generator, validation_generator):

    inp = keras.Input(shape = (224, 224, 3))
    backbone = DenseNet121(input_tensor=inp,
                        weights="imagenet",
                        include_top=False)

    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.2)(x)
    outp = Dense(120, activation="softmax")(x)

    model = Model(inp, outp)

    print(model.summary())

    for layer in model.layers[:-6]:
        layer.trainable = False
    
    model.compile(optimizer= 'RMSprop',
              loss="categorical_crossentropy",
              metrics=["accuracy"])

    model_check = ModelCheckpoint('Dog_Breed_Classifier.h5', 
                                monitor='val_acc', 
                                verbose=1, 
                                save_best_only=True, 
                                mode='max', 
                                save_weights_only=True)

    history = model.fit_generator(generator=train_generator, 
                                steps_per_epoch=len(train_generator), 
                                validation_data=validation_generator, 
                                validation_steps=len(validation_generator),
                                epochs=10,
                                callbacks=[model_check])
    return model, history


#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
#                                   patience=1, verbose=1, mode='min',
#                                   min_delta=0.0001, cooldown=2, min_lr=1e-7)

#early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=5)

if __name__ == "__main__":
    model, history = keras_model()
    model.save(os.path.join(os.getcwd(), 'Dog_Breed_Classifier_New.h5'))
    with open('history_model.pkl', 'wb') as hist:
        pickle.dump(history, hist)




