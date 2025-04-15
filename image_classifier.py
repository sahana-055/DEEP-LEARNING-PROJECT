import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load Pre-trained Model
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Load and Preprocess Image
img_path = "sample.jpg"  # Replace with your image path
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))  # Resize for MobileNetV2
img_array = np.expand_dims(img, axis=0)
img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

# Make Prediction
predictions = model.predict(img_array)
decoded_preds = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]

# Display Image & Predictions
plt.imshow(img)
plt.axis("off")
plt.title(f"Prediction: {decoded_preds[0][1]} ({decoded_preds[0][2]*100:.2f}%)")
plt.show()
