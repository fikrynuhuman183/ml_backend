from dotenv import load_dotenv
import os
from PIL import Image
import tensorflow as tf
import numpy as np

img_height = 224
img_width = 224

load_dotenv()

def nnPredict(image_path: str):
    """
    Predict image classification using the H5 model.
    
    Args:
        image_path (str): Path to the image file
    
    Returns:
        numpy.ndarray: Prediction probabilities for [benign, healthy, opmd]
    """
    model = tf.keras.models.load_model(os.getenv('H5_FILE_PATH'))
    img = Image.open(image_path)
    img = img.resize((img_height, img_width))
    img_array = np.reshape(img, (1, img_height, img_width, 3))
    pred = model.predict(img_array)
    return pred
