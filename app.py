import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Input

@st.cache_data
def read_data():
    folder_path = 'images'
    labels = os.listdir(folder_path)
    n = 1000
    X = None
    y = None
    for i in range(len(labels)):
        subfolder = os.listdir(os.path.join(folder_path, labels[i]))
        imgs = [Image.open(os.path.join(folder_path, labels[i], img)) for img in subfolder[:n]]
        imgs = [np.array(img, dtype=float) for img in imgs]
        imgs = np.stack(imgs)
        # print(labels[i], imgs.shape)
        X = imgs if X is None else np.concatenate((X, imgs))
        if y is None:y = [i] * n
        else:y.extend([i] * n)
    
    y = np.array(y)
    print(X.shape, y.shape)
    return X,y,np.array(labels)

def preprocess(X, y, test_size):
    X /= 255
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
    y_train_ohe = to_categorical(y_train, num_classes=26)
    y_test_ohe = to_categorical(y_test, num_classes=26)
    return X_train, X_test, y_train_ohe, y_test_ohe

def train(X, y, epochs):
    model = Sequential()
    model.add(Input(shape=X.shape[1:]))
    model.add(Flatten())
    model.add(Dense(26, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
    history = model.fit(X, y, epochs = epochs, verbose=1)
    model.save('model.h5')
    return model, history

def train():
    col1, col2 = st.columns(2)
    with col1:
        epochs = st.number_input('Epochs', 10)
    with col2:
        test_size = st.number_input('Test size', min_value=.05, max_value=.5, value=0.1, step=.05)
    X,y,labels = read_data()
    if st.button('Train'):
        with st.spinner('Training...'):
            X_train, X_test, y_train_ohe, y_test_ohe = preprocess(X, y, test_size)
            model,history = train(X_train, y_train_ohe, epochs)
            _, accuracy = model.evaluate(X_test, y_test_ohe)
            fig, _ = plt.subplots(1,2)
            fig.set_figheight(2)
            plt.subplot(1,2,1)
            plt.title('Loss')
            plt.plot(history.history['loss'])
            plt.subplot(1,2,2)
            plt.title('Accuracy')
            plt.plot(history.history['accuracy'])
            st.pyplot(fig)
            st.success(f'Done. Accuracy on test set: {round(accuracy,2)}')

def inference():
    pass

def main():
    tab1, tab2 = st.tabs(('Train', 'Inference'))
    with tab1:
        train()
    with tab2:
        inference()

main()