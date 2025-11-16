import numpy as np
import tensorflow as tf

class DataCleaner:
    def __init__(self):
        pass
    
    def normalize_images(self,images):
        return images.astype('float32')/255.0
    
    def one_hot_encode_labels(self,labels):
        return tf.keras.utils.to_categorical(labels,10)