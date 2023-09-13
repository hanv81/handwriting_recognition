import os
import time
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stqdm import stqdm
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Input, Conv2D, MaxPool2D, Activation, BatchNormalization

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
    X_train = X_train[..., None]
    num_classes = max(y) + 1
    X_train, X_test, y_train, y_test = train_test_split(X_train, y, test_size=test_size, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, stratify=y_train)
    y_train_ohe = to_categorical(y_train, num_classes=num_classes)
    y_test_ohe = to_categorical(y_test, num_classes=num_classes)
    y_val_ohe = to_categorical(y_val, num_classes=num_classes)
    return X_train, X_test, X_val, y_train_ohe, y_test_ohe, y_val_ohe

def create_model(input_shape, cnn_blocks, mlp_layers, use_batchnorm, use_bias, num_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))
    for n_filters, kernel_size in cnn_blocks:
        model.add(Conv2D(n_filters, kernel_size, padding='same', kernel_initializer='he_normal', use_bias=use_bias))
        if use_batchnorm:
            model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPool2D())
    model.add(Flatten())
    for node in mlp_layers:
        model.add(Dense(node, kernel_initializer='he_normal', use_bias=use_bias))
        if use_batchnorm:
            model.add(BatchNormalization())
        model.add(Activation('relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
    model.summary()
    return model

def train(model, X_train, X_val, y_train_ohe, y_val_ohe, epochs):
    t = time.time()
    history = model.fit(X_train, y_train_ohe, epochs = epochs, validation_data=(X_val, y_val_ohe), verbose=1)
    t = int(time.time()-t)
    model.save('model.h5')
    return model, history, t

def visualize_history(history):
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Loss', 'Accuracy'))
    fig.add_trace(go.Scatter(y=history.history['loss'], mode='lines', name='Train Loss'), row=1, col=1)
    fig.add_trace(go.Scatter(y=history.history['val_loss'], mode='lines', name='Val Loss'), row=1, col=1)

    fig.add_trace(go.Scatter(y=history.history['accuracy'], mode='lines', name='Train Accuracy'), row=1, col=2)
    fig.add_trace(go.Scatter(y=history.history['val_accuracy'], mode='lines', name='Val Accuracy'), row=1, col=2)

    fig.update_xaxes(title_text="Epochs", row=1, col=1)
    fig.update_xaxes(title_text="Epochs", row=1, col=2)
    st.plotly_chart(fig)

def main():
    st.set_page_config(
        page_title="Training",
        page_icon="💻",
    )
    tabs = st.tabs(('Dataset', 'Train'))
    with tabs[0]:
        cols = st.columns(2)
        with cols[0]:
            n = st.number_input('Number of samples per class', value=500, min_value=100, max_value=20000, step=500)
        with cols[1]:
            uploaded_file = st.file_uploader('Upload Dataset', type=['zip'])
        X, y, labels = read_data(n)
    with tabs[1]:
        cols = st.columns(4)
        with cols[0]:
            epochs = st.number_input('Epochs', min_value=5, max_value=100, value=10, step=5)
        with cols[1]:
            test_size = st.number_input('Test size', min_value=.1, max_value=.5, value=0.2, step=.05)
        with cols[2]:
            num_of_mlp = st.number_input('Number of hidden layers', min_value=0)
        with cols[3]:
            num_of_cnn_block = st.number_input('Number of CNN blocks', min_value=0)
        use_batchnorm = st.checkbox('Batch-Normalization')
        use_bias = st.checkbox('Use bias')

        mlp_layers = []
        if num_of_mlp > 0:
            cols = st.columns(num_of_mlp)
            for i in range(num_of_mlp):
                with cols[i]:
                    mlp_layers.append(st.selectbox(f'Layer {i+1} nodes', options=[2,4,8,16,32,64,128,256,512,1024], index=2))

        cnn_blocks = []
        if num_of_cnn_block > 0:
            cols = st.columns(num_of_cnn_block)
            for i in range(num_of_cnn_block):
                with cols[i]:
                    st.text(f'Block {i+1}')
                    n_filters = st.selectbox('Number of filters', options=[4,8,32,64,128,512], key=f'filters{i}')
                    kernel_size = st.selectbox('Kernel size', options=[3,5,7,9,11], key=f'size{i}')
                    cnn_blocks.append((n_filters, kernel_size))

        model = create_model(X[..., None].shape[1:], cnn_blocks, mlp_layers, use_batchnorm, use_bias, y.max()+1)
        st.write('Total params:', model.count_params())
        if st.button('Train'):
            X_train, X_test, X_val, y_train_ohe, y_test_ohe, y_val_ohe = preprocess(X, y, test_size)
            with st.spinner('Training...'):
                model, history, t = train(model, X_train, X_val, y_train_ohe, y_val_ohe, epochs)
                _, test_acc = model.evaluate(X_test, y_test_ohe)
                _, train_acc = model.evaluate(X_train, y_train_ohe)
            st.success(f'Training time: {t}s. Train accuracy: {round(train_acc*100,2)}%. Test accuracy: {round(test_acc*100,2)}%')
            visualize_history(history)

main()