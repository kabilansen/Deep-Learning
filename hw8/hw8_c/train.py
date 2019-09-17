import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
import numpy as np
import os
from keras.utils import multi_gpu_model
# os.environ["CUDA_VISIBLE_DEVICES"]="1";  
from PIL import Image
from skimage.transform import resize
import sys

#
from keras.utils import np_utils, generic_utils, to_categorical
from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet121
from keras import backend as K
import keras
from keras.layers import merge, Input
from keras.layers import Dense, Activation, Flatten
from keras.models import Model
from sklearn.utils import shuffle
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import optimizers
# K.set_image_dim_ordering('th')
#########
# Read from directory
#########

def read_data(folder):
    train_test_data = ImageDataGenerator(
            rescale=1./255)


    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_test_data.flow_from_directory(
            folder+'/train',
            target_size=(100, 100),
            batch_size=32,
            class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
            folder+'/val',
            target_size=(100,100),
            batch_size=32,
            class_mode='categorical')
    return train_generator, validation_generator

  
  
  
train_generator, validation_generator = read_data(sys.argv[1])
X_train = train_generator[0]
Y_train = train_generator[1]
image_input = Input(shape=(100,100,3))
model = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=image_input, pooling=None,classes=2)

for layer in model.layers:
    layer.trainable = False

#Adding custom Layers 
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.25)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(2, activation="softmax")(x)

model_final = Model(input = model.input, output = predictions)
# model_final.save(sys.argv[2])

opt = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model_final.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
              metrics=['accuracy'])



def train():
    model_final.fit_generator(train_generator,
              epochs=5,
              validation_data = validation_generator)
    model_final.save(sys.argv[2])

train()
