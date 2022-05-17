import torchvision.models as models
from torchvision.datasets.utils import download_url
import json


class Vgg16:

    def __init__(self):
        self.model = models.vgg16(pretrained=True)
        self.name = "vgg"

        download_url("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json", ".",
                     "imagenet_class_index.json")
        with open("imagenet_class_index.json", "r") as h:
            self.labels = json.load(h)

    def train(self):
        self.model.eval()
        return self.model

    def predict(self, img):
        predictions = self.model(img)
        return predictions


class AlexNet:

    def __init__(self):
        self.model = models.vgg16(pretrained=True)
        self.name = "alex"

        download_url("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json", ".",
                     "imagenet_class_index.json")
        with open("imagenet_class_index.json", "r") as h:
            self.labels = json.load(h)

    def train(self):
        self.model.eval()
        return self.model

    def predict(self, img):
        predictions = self.model(img)
        return predictions
