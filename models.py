import torchvision.models as models
from torchvision.datasets.utils import download_url
import json


class Vgg16:

    def __init__(self):
        self.model = models.vgg16(pretrained=True)
        self.name = "vgg"
        self.ce_layer_name = 'features_29'



        download_url("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json", ".",
                     "data/imagenet_class_index.json")
        with open("data/imagenet_class_index.json", "r") as h:
            self.labels = json.load(h)

    def train(self):
        self.model.eval()
        return self.model

    def predict(self, img):
        predictions = self.model(img)
        return predictions

    def __call__(self, x): 
        return self.model.predict(x)


class AlexNet:

    def __init__(self):
        self.model = models.alexnet(pretrained=True)
        self.name = "alexnet"
        self.ce_layer_name = "features_11"


        download_url("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json", ".",
                     "data/imagenet_class_index.json")
        with open("data/imagenet_class_index.json", "r") as h:
            self.labels = json.load(h)

    def train(self):
        self.model.eval()
        return self.model

    def predict(self, img):
        predictions = self.model(img)
        return predictions

    def __call__(self, x): 
        return self.model.predict(x)
