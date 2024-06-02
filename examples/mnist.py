import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.optimizers import Adam
from keras.datasets import mnist
import numpy as np
from core.data_selector import DataSelector

class CustomModel(tf.keras.Model):
    def __init__(self, D_in, H1, H2, D_out):
        super(CustomModel, self).__init__()
        self.cl1 = tf.keras.layers.Dense(H1, activation='relu', input_shape=(D_in,))
        self.cl2 = tf.keras.layers.Dense(H1, activation='relu')
        self.cl3 = tf.keras.layers.Dense(H2, activation='relu')
        self.fc1 = tf.keras.layers.Dense(D_out, activation='softmax')

    def call(self, inputs):
        x = self.cl1(inputs)
        x = self.cl2(x)
        x = self.cl3(x)
        x = self.fc1(x)
        return x

    def inspector_out(self, inputs):
        x = self.cl1(inputs)
        x = self.cl2(x)
        x = self.cl3(x)  
        return x

def mnist_sample():
    
    epochs=10
    n_components_pca = 20
    update_period_in_epochs=5
    
    (X_train, y_train), (X_val, y_val) = mnist.load_data()
    
   
    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0
    
    
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_val = X_val.reshape((X_val.shape[0], -1))
    
    
    
    enc = OneHotEncoder(sparse_output=False)
    y_train_encoded = enc.fit_transform(y_train.reshape(-1, 1))
    y_val_encoded = enc.transform(y_val.reshape(-1, 1))
    

    
    
    model_mnist = CustomModel(784, 100, 250, 10)
    
   
    learning_rate = 0.001
    adam_optimizer = Adam(learning_rate=learning_rate)
    model_mnist.compile(optimizer=adam_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    data_selector = DataSelector(X_train, y_train_encoded, int(epochs* 0.7), update_period_in_epochs,n_components_pca)
    
 
    for epoch in range(epochs):
        X_tr_filtered, y_tr_filtered = data_selector.get_train_data(epoch=epoch, model=model_mnist, outs_posibilities=list(np.unique(y_train)))
        model_mnist.fit(X_tr_filtered, y_tr_filtered, epochs=1, batch_size=32, verbose=1, validation_data=(X_val, y_val_encoded))
    
    
    data_removed = data_selector.get_removed_data()
    print(data_removed)
    
if __name__ == "__main__":
    mnist_sample()
