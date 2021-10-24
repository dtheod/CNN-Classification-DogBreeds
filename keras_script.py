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
from typing import Union
import time
import sys
sys.setrecursionlimit(3000)

TRAINING_DIR = os.path.join(os.getcwd(), "archive" ,"Training_path")
VALIDATION_DIR = os.path.join(os.getcwd(), "archive","Testing_path")

def time_decorator(func):
    def wrapper(*args, **kwargs):
        from time import time
        start = time()
        result = func(*args, **kwargs)
        print(f'Time needed {time() - start} ms')
        return result
    return wrapper

@time_decorator
def ImageGenerators(training_dir:str, validation_dir:str) -> Union[object, object]:

    #Using the built-in class of the ImageDataGenerator
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

    # No augmentation on the validation data just dividing by 255
    test_datagen = ImageDataGenerator(
        rescale = 1./255.
    )

    #Declare the training data generator using the directory path, target size of the
    #image and batch size
    train_generator = training_datagen.flow_from_directory(
        training_dir,
        target_size=(224,224),
        class_mode='categorical',
        batch_size=32
    )
    
    #Declare the validation data generator using the directory path, target size of the
    #image and batch size
    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(224,224),
        class_mode='categorical',
        batch_size=32
    )

    #Extracting the class names of each class as a dictionary
    label_map = (train_generator.class_indices)

    #Saving the dictionary as a pickle object
    with open(os.path.join(os.getcwd(),'artifacts/label_classes.pkl'), 'wb') as pick:
        pickle.dump(label_map, pick)

    return train_generator, validation_generator


@time_decorator
def keras_model(train_generator:object, validation_generator:object) -> Union[object,object]:

    #Define the input shape of the image
    inp = keras.Input(shape = (224, 224, 3))

    #Initialise the DenseNet121 classifier
    densenet = DenseNet121(input_tensor=inp,
                        weights="imagenet",
                        include_top=False)

    #Expanding the network with our own layers
    x = densenet.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.2)(x)
    outp = Dense(120, activation="softmax")(x)

    model = Model(inp, outp)

    print(model.summary())

    #Putting last 6 layers training into false
    for layer in model.layers[:-6]:
        layer.trainable = False
    
    # Define the optimiser and loss function and metric
    model.compile(optimizer= 'RMSprop',
              loss="categorical_crossentropy",
              metrics=["accuracy"])

    model_check = ModelCheckpoint('Dog_Breed_Classifier.h5', 
                                monitor='val_acc', 
                                verbose=1, 
                                save_best_only=True, 
                                mode='max', 
                                save_weights_only=True)

    # Fitting the model
    history = model.fit_generator(generator=train_generator, 
                                steps_per_epoch=len(train_generator), 
                                validation_data=validation_generator, 
                                validation_steps=len(validation_generator),
                                epochs=1,
                                callbacks=[model_check])
    return model, history


#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
#                                   patience=1, verbose=1, mode='min',
#                                   min_delta=0.0001, cooldown=2, min_lr=1e-7)

#early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=5)

if __name__ == "__main__":

    train_generator, validation_generator = ImageGenerators(TRAINING_DIR, VALIDATION_DIR)
    model, history = keras_model(train_generator, validation_generator)
    model.save(os.path.join(os.getcwd(), 'artifacts/Dog_Breed_Classifier_New1.h5'))
    with open(os.path.join(os.getcwd(),'artifacts/history_model.pkl'), 'wb') as hist:
        pickle.dump(history, hist)




