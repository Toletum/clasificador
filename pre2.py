import tensorflow as tf

from keras.models import load_model

import numpy as np

import pickle

img_height = 231
img_width = 113

class_names = pickle.load(open('class_names.obj', 'rb'))

model = load_model('./model_saved2.h5')

img = tf.keras.utils.load_img(
    "/home/toletum/clasificador/t_data/train/profile/002.jpg", target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to '{}' with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
