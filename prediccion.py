import numpy as np

from keras.models import load_model
from keras_preprocessing.image import load_img

model = load_model('model_saved.h5')

"""
image = load_img('v_data/test/planes/5.jpg', target_size=(224, 224))
img = np.array(image)
img = img / 255.0
img = img.reshape(1,224,224,3)
label = model.predict(img)
print("Predicted Class (0 - Cars , 1- Planes): ", label[0][0])
"""
