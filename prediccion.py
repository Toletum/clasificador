import numpy as np
import tensorflow as tf

from keras_preprocessing.image import load_img

model = tf.keras.models.load_model('./model_saved2.h5')


image = load_img('t_data/test/location/001.jpg', target_size=(113, 231))
img = np.array(image)
img = img / 255.0
img = img.reshape(1, 113, 231, 3)
label = model.predict(img)
print("Predicted Class (0 - Cars , 1- Planes): ", label[0][0])
