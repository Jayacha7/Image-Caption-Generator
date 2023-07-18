import streamlit as st
import os
import numpy as np
import pickle
from tqdm.notebook import tqdm
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer

st.header("Image Caption generator")
uploaded_file = st.file_uploader("Choose a image file", type="jpg")

def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


if uploaded_file is not None:
    st.image(uploaded_file)
    model = keras.models.load_model('best_model.h5')
    vgg_model = VGG16()
    vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
    image = load_img(uploaded_file, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = vgg_model.predict(image, verbose=0)

    tokenizer = Tokenizer()
    with open(os.path.join("", 'allcap.pkl'), 'rb') as f:
        all_captions = pickle.load(f)
    tokenizer.fit_on_texts(all_captions)
    sqw =predict_caption(model, feature, tokenizer, 35)
    st.write(sqw.replace("startseq","").replace("endseq",""))
