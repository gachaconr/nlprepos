import urllib.request
import zipfile
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_height = 113
img_width = 187

model = tf.keras.models.load_model('cablemodel3.keras')

dirp = "/home/gachaconr/Desktop/tf/tensorflow_files/imageclassification/cable_green_resized/"
ok = "pass"
nok = "fail"
Tok = 0
Tnok = 0

for i in range(1,126):
	print(i)
	path = dirp + ok + "/resized" + ok + str(i) + ".jpg"
	img = tf.keras.utils.load_img( path, target_size=(img_height, img_width) )
	img_array = tf.keras.utils.img_to_array(img)
	img_array = tf.expand_dims(img_array, 0) # Create a batch
	predictions = model.predict(img_array)
	if (predictions[0] > 0.5):
		Tok = Tok + 1
	else:
		print("NO OK")
		print(predictions[0])

	path = dirp + nok + "/resized" + nok + str(i) + ".jpg"
	img = tf.keras.utils.load_img( path, target_size=(img_height, img_width) )
	img_array = tf.keras.utils.img_to_array(img)
	img_array = tf.expand_dims(img_array, 0) # Create a batch
	predictions = model.predict(img_array)
	if (predictions[0] < 0.5):
		Tnok = Tnok + 1
	else:
		print("NO OK")
		print(predictions[0]) 

print("Tok = " + str(Tok))
print("Tnok = " + str(Tnok))


