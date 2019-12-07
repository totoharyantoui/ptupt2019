


import sys
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

import keras

from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation, merge
from keras.layers import MaxPooling2D,Dense,Conv2D,MaxPooling2D,Flatten,AveragePooling2D,Dropout,BatchNormalization,Activation
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint

#import keras utility for multiple GPU
#---------------------------------------
from keras.utils import multi_gpu_model
#----------------------------------------
import tensorflow as tf





#from keras.layers.convolutional import Convolution2D, MaxPooling2D

from keras import callbacks
import time
import json


start = time.time()



DEV = False
argvs = sys.argv
argc = len(argvs)



if argc > 1 and (argvs[1] == "--development" or argvs[1] == "-d"):

  DEV = True

if DEV:

 epochs = 2

else:

  epochs = 1000



train_data_path = '/home/toto/ResNet/GlandAsli/training/'
validation_data_path = '/home/toto/ResNet/GlandAsli/validation/'
	


"""

Parameters

"""

img_width, img_height = 224,224
batch_size = 32
samples_per_epoch = 1500
validation_steps = 100

nb_filters1 = 32
nb_filters2 = 64
nb_filters3 = 96

conv1_size = 3
conv2_size = 2

pool_size = 2
classes_num = 2

lr = 0.0001


#first model

model = Sequential()
model.add(Conv2D(nb_filters1, kernel_size=[7,7], padding ="same", input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Conv2D(nb_filters1, kernel_size=[5,5], padding ="same", input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Conv2D(nb_filters1, kernel_size=[7,7], padding = "same", input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Conv2D(nb_filters1, kernel_size=[5,5], padding = "same", input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))


#second model

model.add(Conv2D(nb_filters2, kernel_size=[7,7], padding = "same", input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Conv2D(nb_filters2, kernel_size=[5,5], padding = "same", input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

model.add(Conv2D(nb_filters2, kernel_size=[7,7], padding = "same", input_shape=(img_width, img_height, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))


#combine model

model.add(Flatten())
	
model.add(Dense(1024,kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01)))
model.add(Activation("relu"))
model.add(Dropout(0.3))

model.add(Dense(512,kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01)))
model.add(Activation("relu"))
model.add(Dropout(0.3))


model.add(Dense(256,kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01)))
model.add(Activation("relu"))
model.add(Dropout(0.3))


model.add(Dense(classes_num, activation='softmax'))
# runn model on multiple GPU
#===============================================================================================================
#parallel_model = multi_gpu_model(model,3)
model.compile(loss='categorical_crossentropy',optimizer=optimizers.Adam(lr=lr),metrics=['accuracy'])

#
filepath = "weight.best.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


#===============================================================================================================

  
print(model.summary())
       
train_datagen = ImageDataGenerator(

    #rescale=1. / 255,
    rescale=None,
    #shear_range=0.2,
    #zoom_range=0.2,
    rotation_range=.3,
    width_shift_range=.15,
    height_shift_range=.15,
    vertical_flip=True,
    horizontal_flip=True)



#test_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=None)



train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')



validation_generator = test_datagen.flow_from_directory(

    validation_data_path,

    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')



"""

Tensorboard log

"""

#log_dir = './tf-log/'

#tb_cb = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)

#cbks = [tb_cb]


history = model.fit_generator(

    train_generator,

    samples_per_epoch=samples_per_epoch,

    epochs=epochs,

    validation_data=validation_generator,

    callbacks=callbacks_list,

    validation_steps=validation_steps)



target_dir = './models/'

if not os.path.exists(target_dir):

  os.mkdir(target_dir)

model.save('./models/model_200_757.h5')

model.save_weights('./models/weights_200_757.h5')

print(history.history.keys())

with open('./models/history_model_200_757.json', 'w') as f:
    json.dump(history.history, f)





#Calculate execution time

end = time.time()

dur = end-start

file = open('./models/runtime_model_200_757.txt','w')
file.write(str(dur))
file.close()




if dur<60:

    print("Execution Time:",dur,"seconds")

elif dur>60 and dur<3600:

    dur=dur/60

    print("Execution Time:",dur,"minutes")

else:

    dur=dur/(60*60)

    print("Execution Time:",dur,"hours")


