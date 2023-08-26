import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def read_data():
    folder_path = 'images'
    labels = os.listdir(folder_path)
    n = 100
    X = None
    y = None
    for i in range(len(labels)):
        subfolder = os.listdir(os.path.join(folder_path, labels[i]))
        imgs = [Image.open(os.path.join(folder_path, labels[i], img)) for img in subfolder[:n]]
        imgs = [np.array(img, dtype=float) for img in imgs]
        imgs = np.stack(imgs)
        print(labels[i], imgs.shape)
        X = imgs if X is None else np.concatenate((X, imgs))
        if y is None:y = [i] * n
        else:y.extend([i] * n)
    
    y = np.array(y)
    print(X.shape, y.shape)
    return X,y,np.array(labels)

def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    X_train /= 255
    X_test /= 255
    y_train_ohe = to_categorical(y_train, num_classes=26)
    y_test_ohe = to_categorical(y_test, num_classes=26)

def main():
    X,y,labels = read_data()
    model = train(X, y)

main()