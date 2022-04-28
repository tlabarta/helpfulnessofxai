import matplotlib.pyplot as plt
from tensorflow import keras
from keras import datasets


class DatasetHandler:
    """Loads, initializes and prints an example from a dataset"""

    def __init__(self, dataset, class_labels):
        # load data
        self.dataset = dataset
        (self.train_images, self.train_labels), \
        (self.test_images, self.test_labels) = dataset.load_data()

        # normalize pixel values
        self.train_images = self.train_images / 255.0
        self.test_images = self.test_images / 255.0

        # class labels
        self.class_labels = class_labels

    def print(self, index):
        plt.imshow(self.train_images[index], cmap=plt.cm.binary)
        plt.xlabel(self.class_labels[self.train_labels[index][0]])
        plt.show()
