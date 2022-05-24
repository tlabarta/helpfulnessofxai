from torchvision import datasets
from torch.utils import data
import torchvision.transforms as transforms
import json
from torchvision.datasets.utils import download_url
import os
import numpy as np
import matplotlib.pyplot as plt


def get_image(path):
    # get the image from the dataloader
    dataset = datasets.ImageFolder(root=path, transform=transform())
    dataloader = data.DataLoader(dataset=dataset, shuffle=False, batch_size=1)  # batch_size=1
    return iter(dataloader)


def transform():
    # transforms.ToTensor() normalizes input data from 0-255 to 0-1
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])

    return transform


def get_files(path):
    images = "images/"
    files = os.listdir(path + images)

    return files


def get_labels():
    # Download class labels from imagenet dataset
    download_url("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json", ".",
                 "data/imagenet_class_index.json")

    with open("data/imagenet_class_index.json", "r") as h:
        labels = json.load(h)
    return labels


def topk_confidence_scores(preds, labels, k):
    """
    :param preds: class predictions
    :param labels: class labels
    :param k: top k confidence scores to be returned
    saves the output graph into
    :return: a list of tuples with axis 0 being confidence scores, axis 1 - corresponding class labels
    """
    # transform predictions to numpy array
    predictions_tensor = preds.detach().cpu().numpy()
    predictions_tensor = np.squeeze(predictions_tensor)
    np.set_printoptions(formatter={'float_kind': '{:f}'.format})

    # sort array in descending order
    sorted_confidence_scores = np.sort(predictions_tensor)[::-1]
    sorted_confidence_scores = sorted_confidence_scores[0:k]

    # extract corresponding class labels
    ind = np.argsort(predictions_tensor)[::-1][0:k]
    sorted_predicted_labels = np.array([])
    for i in range(0, k):
        str_label = str(ind[i])
        sorted_predicted_labels = np.append(sorted_predicted_labels, labels[str_label][1])

    # list of scores and class labels
    confidence_scores = np.vstack((sorted_confidence_scores, sorted_predicted_labels)).T

    # plot scores
    plt.barh(sorted_predicted_labels[::-1], sorted_confidence_scores[::-1], height=0.5)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig("results/confidence_scores/scores.jpg")
    return confidence_scores
