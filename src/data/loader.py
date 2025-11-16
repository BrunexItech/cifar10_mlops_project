import tensorflow as tf
import numpy as np


class CIFAR10DataLoader:
    def __init__(self):
        self.class_names=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
        
    def load_data(self):
        (x_train,y_train),(x_test,y_test)=tf.keras.datasets.cifar10.load_data()
        return (x_train,y_train),(x_test,y_test)