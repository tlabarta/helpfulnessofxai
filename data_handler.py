import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torchvision.transforms as transforms
from torch.utils import data
from torchvision.datasets.utils import download_url
from torchvision import datasets


def get_image(path):
    # get the image from the dataloader
    dataset = datasets.ImageFolder(root=path, transform=transform())
    dataloader = data.DataLoader(dataset=dataset, shuffle=False)  # batch_size=1

    return iter(dataloader)


def transform():
    """
    Preprocessing as defined in https://github.com/pytorch/examples/blob/main/imagenet/main.py for 
    validiation data
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    return transform


def get_files(path):
    images = "images/"
    files = os.listdir(path + images)

    return files


def get_labels(path=None):
    # Download class labels from imagenet dataset
    if not path:
        download_url("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json", ".",
                     "data/imagenet_class_index.json")

        with open("data/imagenet_class_index.json", "r") as h:
            labels = json.load(h)

    else:
        download_url("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json", ".",
                     path)

        with open(path, "r") as h:
            labels = json.load(h)

    return labels


def get_question_image(testset_path, img_idx, labels):
    img_folder = datasets.ImageFolder(root=testset_path)
    img_path = img_folder.imgs[img_idx][0]
    pil_img = img_folder.loader(img_path)
    img_org_np = np.expand_dims(np.array(pil_img.resize((224, 224))), 0)
    img_name = img_path.split(os.sep)[-1]
    # preprocessing
    img_prep_torch = transform()(pil_img)
    img_prep_torch = img_prep_torch.unsqueeze(0)
    # extract correct class
    class_idx_true_str = img_path.split(os.sep)[-2]
    img_label_true = labels[class_idx_true_str][1]

    return img_org_np, img_prep_torch, img_name, img_label_true


def get_questionnaires(path):
    with open(path, 'rb') as f:
        questionnaires_list = pickle.load(f)

    return questionnaires_list


def get_figure_from_img_array(image_np, title):
    plt.imshow(image_np)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    fig = plt.gcf()
    plt.close()

    return fig
