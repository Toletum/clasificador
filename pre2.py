import tensorflow as tf

from keras.models import load_model

import numpy as np

img_height = 231
img_width = 113

class_names = ['location', 'match', 'menu', 'profile', 'thanks']

model = load_model('./model_saved2.h5')

img = tf.keras.utils.load_img(
    "/home/toletum/clasificador/t_data/train/location/002.jpg", target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to '{}' with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
