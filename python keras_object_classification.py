#=================================================================
# 7/12/21
# Object classification using Keras and specifically MobileNetV2
# Melih Aslan
# maslanky@gmail.com

# Usage:
# Adjust the parameters num_classes , SIZE_h, SIZE_w, train_batchsize for your applications
# Define the training, validation, and testing folders. Each class images should be subfoldered like 0, 1, ..., num_classes-1.
# Define augmentation variables in Data Generators
# Adjust variables in "base_model = MobileNetV2" based on your needs or leave it as it is.

# Run:
# type in your command window: "python keras_object_classification.py"
#=================================================================


# HELP SOURCE
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# https://www.learnopencv.com/keras-tutorial-fine-tuning-using-pre-trained-models/
#https://github.com/keras-team/keras/issues/7904


import numpy as np
import keras
from numpy import argmax
from keras import models
from keras import layers
from keras import optimizers
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import Xception, preprocess_input
from keras.models import Model
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.optimizers import Adagrad
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint


# Define Variables	
num_classes = 2 
SIZE_h = 64
SIZE_w = 64

# Change the batchsize according to your system RAM
train_batchsize = 32
val_batchsize = 1

# Define Directories
train_images = 'data/combined_2/train'
val_images = 'data/64x64_plus_plus'
test_images = 'data/64x64_plus_plus'


# Define Data Generators
# TRAIN
train_datagen = ImageDataGenerator(
      rescale=1./255, 
      featurewise_center=True, 
      featurewise_std_normalization=True,
      rotation_range=360,
      width_shift_range=0.2,
      height_shift_range=0.2,
      brightness_range = (-30,30),
      horizontal_flip=True,
      fill_mode='nearest',
      vertical_flip=True,
      shear_range=0.05, 
      zoom_range=[0.7,1.3]      
      samplewise_center=True,
      featurewise_center=True,
      zca_whitening=True
	  )
      
train_dir = train_images 

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(SIZE_h, SIZE_w),
        batch_size=train_batchsize,
        class_mode='categorical',
        interpolation='bilinear',
        shuffle=True
		#,
		#color_mode = "grayscale"
		)

        
# VALIDATION      
validation_datagen = ImageDataGenerator(
                        rescale=1./255, 
                        featurewise_center=True, 
                        featurewise_std_normalization=True
                        featurewise_center=True,
                        samplewise_center=True
                        zca_whitening=True
                        )

validation_dir = val_images
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(SIZE_h, SIZE_w),
        batch_size=val_batchsize,
        class_mode='categorical',
        interpolation='bilinear',
        shuffle=False
		#,
		#color_mode = "grayscale"
		)

# TEST        
test_datagen = ImageDataGenerator(
                    rescale=1./255,
                    featurewise_center=True, 
                    featurewise_std_normalization=True
                    featurewise_center=True,
                    samplewise_center=True,
                    zca_whitening=True
                    )

test_dir = test_images
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(SIZE_h, SIZE_w),
        batch_size=1,
        class_mode='categorical',
        interpolation='bilinear',
        shuffle=False
		#,
		#color_mode = "grayscale"
		)	




# Define the model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(SIZE_h, SIZE_w, 3), alpha = 0.75) # default weights = 'imagenet'


# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x) # 1024 default

# and a logistic layer 
predictions = Dense(num_classes, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)



# Define the layers that we want to train/freeze
# at this point, we can select which layers will be trained.
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train all layers
for layer in model.layers[:1]:
   layer.trainable = False

for layer in model.layers[0:]:
   layer.trainable = True



# Define hyperparameters for the model
# optimization parameters.
# the best network was obtained using the following optimization: model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss='categorical_crossentropy', metrics=['accuracy'])
# I keep some candidates for future use.
model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss='categorical_crossentropy', metrics=['accuracy'])


# Show a summary of the model. Check the number of trainable parameters
model.summary()	

# Define Checkpoint to save weights
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


# Train the model
history = model.fit_generator(
      train_generator,
      #test_generator,
      steps_per_epoch=train_generator.samples/train_generator.batch_size ,
      epochs=22,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples/validation_generator.batch_size,
      callbacks=callbacks_list, 
      verbose=1)   
 
 
#######################################    
# RESULT ANALYSIS PART
#######################################    

# Get the filenames from the generator
fnames = validation_generator.filenames
 
# Get the ground truth from generator
ground_truth = validation_generator.classes
 
# Get the label to class mapping from the generator
label2index = validation_generator.class_indices
 
# Getting the mapping from class index to class label
idx2label = dict((v,k) for k,v in label2index.items())

validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(SIZE_h, SIZE_w),
        batch_size=1,
        class_mode='categorical',
        shuffle=False)	

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(SIZE_h, SIZE_w),
        batch_size=1,
        class_mode='categorical',
        shuffle=True)

		
# Get the predictions from the model using the generator
predictions = model.predict_generator(validation_generator, steps=(validation_generator.samples/validation_generator.batch_size),verbose=1)
predicted_classes = np.argmax(predictions,axis=-1)


# saving results 
errors = np.where(predicted_classes != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors),validation_generator.samples))

# writing the misread paths in validation
f= open("missreads.txt","w+")
# Show the errors
for i in range(len(errors)):

    f.write(fnames[errors[i]])
    pred_class = np.argmax(predictions[errors[i]])
    pred_label = idx2label[pred_class]	
    f.write("predicted label is = .... ")
    f.write(pred_label)	
    f.write("\n")

#=========================================================================	

