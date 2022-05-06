import tensorflow as tf
from tensorflow import keras
from keras import datasets, preprocessing
from dataset_handler import DatasetHandler
from keras.preprocessing import image
import numpy as np

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

def load_image(img_path):
    # each model requires its own imagesize, check docs before using on another model
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = img.reshape(224, 224, 3)
    print(img.shape)
    return img
    

def predict(model, img):
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.xception.preprocess_input(img)
    print(img.shape)
    preds = model.predict(img)
    # print top 3 predictions
    print('Predicted:', tf.keras.applications.xception.decode_predictions(preds, top=3)[0])
    return preds