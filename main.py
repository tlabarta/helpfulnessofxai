import tensorflow as tf
from tensorflow import keras
from keras import datasets, preprocessing
from dataset_handler import DatasetHandler
from keras.preprocessing import image
import numpy as np


def dataset_caller():
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    index = 23456
    dataset_1 = DatasetHandler(datasets.cifar10, class_names)
    dataset_1.print(index)


# TODO move to a separate class
def train():
    # basic out-of-the-box Xception cnn implementation, pretrained on imagenet dataset
    # for playing around with it and letting it classify random wikimedia pictures
    model = tf.keras.applications.Xception(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation="softmax",
    )
    return model


def predict(model, img_path):
    # each model requires its own imagesize, check docs before using on another model
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.xception.preprocess_input(x)

    preds = model.predict(x)
    # print top 3 predictions
    print('Predicted:', tf.keras.applications.xception.decode_predictions(preds, top=3)[0])


model = train()
predict(model, 'images/lion.jpg')
predict(model, 'images/sealion.jpg')
predict(model, 'images/sealions_on_the_beach.jpg')

#dataset_caller()
#this is a test