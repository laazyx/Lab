import tensorflow as tf
import streamlit as st
from tensorflow import keras
from keras.models import load_model
from PIL import Image
import io
import time
from PIL import Image, ImageOps
import numpy as np
import cnn_model
import os

st.markdown("""
    <style>
    .reportview-container {
        background: url("https://thedatacompany.com/wp-content/uploads/2020/06/June-Blog-1500x960.jpg")
    }
   .sidebar .sidebar-content {
        background: url("")
    }
    </style>
    """,
            unsafe_allow_html=True
            )

fas_data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fas_data.load_data()

#cnn_model = load_model("C:/Users/yahya/PycharmProjects/pythonProject/Lab/TensorFlow/cnn_model")
cnn_model = load_model("cnn_model")

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

add_selectbox = st.sidebar.selectbox(
    'select the model for classification',
    ('Welcome', 'CNN')
#comment
)

if add_selectbox == 'Welcome':
    st.title("WELCOME TO Fashion MNIST")

# if add_selectbox == 'About':
# def model_summary():
#     img = Image.open(r"summary.png")
#     st.image(img)
#
#
# def model_training():
#     img = Image.open(r"training.png")
#     st.image(img)
#
# def model_plot():
#     img = Image.open(r"plot.png")
#     st.image(img)
#
#
# if st.button('CNN Model Summary'):
#     model_summary()
# if st.button('CNN Model training'):
#     model_training()
# if st.button('CNN Model plot'):
#     model_plot()

if add_selectbox == 'CNN':
    st.title("Fashion MNIST using CNN")
    file_uploader = st.file_uploader('Upload cloth Image for Classification:')
    st.set_option('deprecation.showfileUploaderEncoding', False)

    if file_uploader is not None:
        image = Image.open(file_uploader)
        text_io = io.TextIOWrapper(file_uploader)
        image = image.resize((180, 180))
        st.image(image, 'Uploaded image:')


        def classify_image(image, model):
            st.write("classifying......")
            img = ImageOps.grayscale(image)

            img = img.resize((28, 28))
            img = np.expand_dims(img, 0)
            img = np.expand_dims(img, 3)
            img = (img / 255.0)

            img = 1 - img

            pred = model.predict(img)

            st.write("The Predicted image is:", class_names[np.argmax(pred)])
            st.write('Prediction probability :{:.2f}%'.format(np.max(pred) * 100))


        st.write('Click her to classify the image')
        if st.button('Classify Image'):
            classify_image(image, cnn_model)
            st.success('Image successfully classified!')
            with st.spinner('Wait for it...'):
                time.sleep(2)
                st.success('Done!')
                st.balloons()
    else:
        st.write("Please select image:")
