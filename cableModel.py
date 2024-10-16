import urllib.request
import zipfile
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_height = 113
img_width = 187

training_dir = '/home/gachaconr/Desktop/tf/tensorflow_files/imageclassification/cable_green_resized/'
train_datagen = ImageDataGenerator(rescale=1/255)
train_generator = train_datagen.flow_from_directory(
	training_dir,
	target_size = (img_height, img_width),
	class_mode = 'binary'
)
convsize = 3
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(16, (convsize,convsize), activation='relu', input_shape=(img_height,img_width,3)),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(32, (convsize,convsize), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(64, (convsize,convsize), activation='relu'),  
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(64, (convsize,convsize), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(64, (convsize,convsize), activation='relu'),  
  tf.keras.layers.MaxPooling2D(2,2),  
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00001,l2=0.00001)),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.summary()
history = model.fit(train_generator, epochs=35)
#model.save('cablemodel4.keras')

print("done")
