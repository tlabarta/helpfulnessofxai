import torch
from torchvision.models import vgg16
from torchvision.datasets.utils import download_url
import json


def train():
    model = vgg16(pretrained=True)
    model.eval()
    return model


def predict(model, img):
    predictions = model(img)
    return predictions


def get_labels():
    # Download class labels from imagenet dataset
    download_url("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json", ".",
                 "imagenet_class_index.json")

    with open("imagenet_class_index.json", "r") as h:
        labels = json.load(h)
    return labels
