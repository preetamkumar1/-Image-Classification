import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model("model.h5")

# Load image
img = image.load_img("test.jpg", target_size=(150,150))
img_array = image.img_to_array(img)/255
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)

if prediction[0] > 0.5:
    print("Dog 🐶")
else:
    print("Cat 🐱")
