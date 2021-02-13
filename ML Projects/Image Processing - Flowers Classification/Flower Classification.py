import numpy as np
import os
import keras
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.optimizers import SGD
from keras.layers import Activation, Dense, GlobalAveragePooling2D, Dropout
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Flatten
import sys

train_directory= sys.argv[1]
model_name = sys.argv[2]

base_model = VGG16(include_top=False, weights="imagenet", input_shape=(224,224,3), pooling=None, classes=10)

x = base_model.output
x = Flatten(name="flatten")(x)

x = Dropout(0.5)(x)
x = Dense(120, activation="relu", name="dense_1")(x)

preds = Dense(5,activation="softmax", name="dense_2")(x)
model = Model(inputs=base_model.input, outputs=preds)

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, data_format="channels_last",validation_split=0.2)
train_datagenerator = train_datagen.flow_from_directory(train_directory , target_size=(224,224), color_mode="rgb", batch_size=32, class_mode="categorical", shuffle=True,subset='training')
validation_datagenerator = train_datagen.flow_from_directory(train_directory , target_size=(224,224), color_mode="rgb", batch_size=32, class_mode="categorical", shuffle=True,subset='validation')

opt = keras.optimizers.SGD(lr=0.001, decay=1e-6)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

step_size = train_datagenerator.n//train_datagenerator.batch_size
validation_steps= validation_datagenerator.n//validation_datagenerator.batch_size

model.fit_generator(generator=train_datagenerator, steps_per_epoch=step_size,validation_data=validation_datagenerator, validation_steps=validation_steps, epochs=25)

model.save(model_name)


