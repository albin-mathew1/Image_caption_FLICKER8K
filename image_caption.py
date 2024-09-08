import numpy as np
from PIL import Image
import os
import string
from pickle import load
from tensorflow.keras.models import load_model
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import concatenate
from keras.layers import add
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.layers import LSTM, Embedding, Dropout
import numpy as np
import matplotlib.pyplot as plt

# Load tokenizer and trained model
tokenizer = load(open("tokenizer.p", "rb"))
model = load_model('model_9.h5')

# Function to extract features from an image using Xception model
def extract_features(filename, model):
    try:
        image = Image.open(filename)
    except:
        print("ERROR: Can't open image! Ensure that image path and extension are correct")
    image = image.resize((299, 299))
    image = np.array(image)
    if image.shape[2] == 4:
        image = image[..., :3]
    image = np.expand_dims(image, axis=0)
    image = image/127.5
    image = image - 1.0
    feature = model.predict(image)
    return feature

# Function to map an integer to a word in the tokenizer's word index
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Function to generate a textual description of an image
def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text

# Load pre-trained Xception model for feature extraction
xception_model = Xception(include_top=False, pooling="avg")

# Function to display image and generated description
def display_image_with_description(img_path, description):
    plt.imshow(Image.open(img_path))
    plt.title("Generated Description: \n" + description)
    plt.axis('off')
    plt.show()

# Image path (specify the path to your image)
img_path = "image.jpg"

# Extract features from the input image
photo = extract_features(img_path, xception_model)

# Generate description for the input image
max_length = 33
description = generate_desc(model, tokenizer, photo, max_length)

# Display the generated description and the input image
display_image_with_description(img_path, description)
