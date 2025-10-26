import numpy as np
import tensorflow as tf

class MNIST_Dataset:
    def __init__(self,):
        self.dataset = tf.keras.datasets.mnist.load_data()
        ((train_values, train_labels), (test_values, test_labels)) = self.dataset


        train_values = (train_values.astype('float32')/255.0).np.reshape(-1, 28 *28)
        test_values = test_values.astype('float32')/255.0.np.reshape(-1, 28*28)
        self.train = (train_values, train_labels)
        self.test = (test_labels, test_labels)

    def get_train_data(self):
        """
        returns train data as a tuple with first Value as normalized picture value from mnist
        and second labels
        """
        return self.train

    def get_test_data(self):
        """
        returns train data as a tuple with first Value as normalized picture value from mnist
        and second labels
        """
        return self.test