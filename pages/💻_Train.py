import os
import time
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from stqdm import stqdm
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Input

DS_PATH = 'dataset'

@st.cache_data
def read_data(n = 1000):
    labels = os.listdir(DS_PATH)
    X = None
    y = None
    # for i in stqdm(range(len(labels))):
    for i in range(len(labels)):
        subfolder = os.listdir(os.path.join(DS_PATH, labels[i]))
        imgs = [Image.open(os.path.join(DS_PATH, labels[i], file)) for file in subfolder[:n]]
        imgs = np.stack([np.array(img, dtype=float) for img in imgs])
        X = imgs if X is None else np.concatenate((X, imgs))
        if y is None:y = [i] * len(imgs)
        else:y.extend([i] * len(imgs))
        print(labels[i], imgs.shape)
    
    y = np.array(y)
    print(X.shape, y.shape)
    return X,y,np.array(labels)

def preprocess(X, y, test_size):
    print(X.shape, y.shape, test_size)
    X_train = X / 255
    num_classes = max(y) + 1
    X_train, X_test, y_train, y_test = train_test_split(X_train, y, test_size=test_size, stratify=y)
    y_train_ohe = to_categorical(y_train, num_classes=num_classes)
    y_test_ohe = to_categorical(y_test, num_classes=num_classes)
    return X_train, X_test, y_train_ohe, y_test_ohe

def train(X, y, nodes, epochs, num_classes):
    model = Sequential()
    model.add(Input(shape=X.shape[1:]))
    model.add(Flatten())
    for node in nodes:
        model.add(Dense(node, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
    model.summary()
    t = time.time()
    history = model.fit(X, y, epochs = epochs, verbose=1)
    t = int(time.time()-t)
    model.save('model.h5')
    return model, history, t

def main():
    st.set_page_config(
        page_title="Training",
        page_icon="💻",
    )
    tabs = st.tabs(('Dataset', 'Train'))
    with tabs[0]:
        n = st.slider('Number of samples per class', min_value=100, max_value=20000, step=200)
        uploaded_file = st.file_uploader('Upload Dataset', type=['zip'])
        X, y, labels = read_data(n)
    with tabs[1]:
        cols = st.columns(4)
        with cols[0]:
            epochs = st.slider('Epochs', min_value=5, max_value=100, value=10, step=5)
        with cols[1]:
            test_size = st.slider('Test size', min_value=.05, max_value=.5, value=0.1, step=.05)
        with cols[2]:
            num_of_mlp = st.number_input('Number of hidden layers', min_value=0)
        with cols[3]:
            num_of_cnn_block = st.number_input('Number of CNN blocks', min_value=0)

        nodes = []
        if num_of_mlp > 0:
            cols = st.columns(num_of_mlp)
            for i in range(num_of_mlp):
                with cols[i]:
                    node = st.selectbox(f'Layer {i+1} nodes', options=[2,4,8,16,32,64,128,256,512,1024], index=2)
                    nodes.append(node)

        if st.button('Train'):
            X_train, X_test, y_train_ohe, y_test_ohe = preprocess(X, y, test_size)
            with st.spinner('Training...'):
                model,history,t = train(X_train, y_train_ohe, nodes, epochs, y.max()+1)
                _, accuracy = model.evaluate(X_test, y_test_ohe)
            st.success(f'Done. Training time: {t}s. Accuracy on test set: {round(accuracy*100,2)}%')
            fig, _ = plt.subplots(1,2)
            fig.set_figheight(2)
            plt.subplot(1,2,1)
            plt.title('Loss')
            plt.xlabel('Epochs')
            plt.plot(history.history['loss'])
            plt.subplot(1,2,2)
            plt.title('Accuracy')
            plt.xlabel('Epochs')
            plt.plot(history.history['accuracy'])
            st.pyplot(fig)
main()