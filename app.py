import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from streamlit_drawable_canvas import st_canvas
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Flatten, Input

DS_PATH = 'dataset'

@st.cache_data
def read_data(n = 1000):
    labels = os.listdir(DS_PATH)
    X = None
    y = None
    for i in tqdm(range(len(labels))):
        subfolder = os.listdir(os.path.join(DS_PATH, labels[i]))
        imgs = [Image.open(os.path.join(DS_PATH, labels[i], file)) for file in subfolder[:n]]
        imgs = np.stack([np.array(img, dtype=float) for img in imgs])
        # print(labels[i], imgs.shape)
        X = imgs if X is None else np.concatenate((X, imgs))
        if y is None:y = [i] * n
        else:y.extend([i] * n)
    
    y = np.array(y)
    # print(X.shape, y.shape)
    return X,y,np.array(labels)

def preprocess(X, y, test_size):
    X /= 255
    num_classes = max(y) + 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
    y_train_ohe = to_categorical(y_train, num_classes=num_classes)
    y_test_ohe = to_categorical(y_test, num_classes=num_classes)
    return X_train, X_test, y_train_ohe, y_test_ohe

def train(X, y, epochs, num_classes):
    model = Sequential()
    model.add(Input(shape=X.shape[1:]))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
    history = model.fit(X, y, epochs = epochs, verbose=1)
    model.save('model.h5')
    return model, history

def training():
    n = st.slider('Number of samples per class', min_value=1000, max_value=20000, step=200)
    X, y, labels = read_data(n)
    epochs = st.slider('Epochs', min_value=5, max_value=50, value=10, step=5)
    test_size = st.slider('Test size', min_value=.05, max_value=.5, value=0.1, step=.05)

    if st.button('Train'):
        with st.spinner('Training...'):
            X_train, X_test, y_train_ohe, y_test_ohe = preprocess(X, y, test_size)
            model,history = train(X_train, y_train_ohe, epochs, y.max()+1)
            _, accuracy = model.evaluate(X_test, y_test_ohe)
            st.success(f'Done. Accuracy on test set: {round(accuracy,2)}')
            fig, _ = plt.subplots(1,2)
            fig.set_figheight(2)
            plt.subplot(1,2,1)
            plt.title('Loss')
            plt.plot(history.history['loss'])
            plt.subplot(1,2,2)
            plt.title('Accuracy')
            plt.plot(history.history['accuracy'])
            st.pyplot(fig)

    return labels, X[0].shape

def inference(labels, input_shape):
    st.subheader('Draw a letter (A-Z) or upload an image')
    col1, col2 = st.columns(2)
    with col1:
        canvas_result = st_canvas(
            stroke_width=8,
            stroke_color='rgb(255, 255, 255)',
            background_color='rgb(0, 0, 0)',
            height=150,
            width=150,
            key="canvas",
        )
    if canvas_result.image_data is not None:img = Image.fromarray(canvas_result.image_data)

    with col2:
        uploaded_file = st.file_uploader('', type=['png','jpg','bmp'])
        if uploaded_file is not None:img = Image.open(uploaded_file)

    if st.button('Predict'):
        model = load_model('model.h5')
        img = img.resize(input_shape)
        img = img.convert('L')
        img = np.array(img, dtype=float)/255
        if uploaded_file is not None: 
            col3, col4, col5 = st.columns(3)
            with col3:
                st.text('Original Image')
                st.image(Image.open(uploaded_file))
            with col4:
                st.text('Grayscale Image')
                inference_image(model, labels, img, input_shape)
            with col5:
                st.text('Invert Grayscale Image')
                inference_image(model, labels, 1-img, input_shape)
        else:
            inference_image(model, labels, img, input_shape, False)

def inference_image(model, labels, img, input_shape, draw_img=True):
    probs = model.predict(img.reshape(-1,*input_shape)).squeeze()*100
    ids = np.argsort(probs)[::-1]
    if draw_img:st.image(img)
    for i in ids[:5]:
        st.write(labels[i], ':', probs[i].round(decimals=2), '%')

def main():
    st.title('HANDWRITING RECOGNITION')
    with st.sidebar:
        labels, input_shape = training()
    inference(labels, input_shape)

main()