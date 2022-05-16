from torchvision import datasets
from torch.utils import data
import torchvision.transforms as transforms
import json
from torchvision.datasets.utils import download_url


def get_image(path):
    # get the image from the dataloader
    dataset = datasets.ImageFolder(root=path, transform=transform())
    dataloader = data.DataLoader(dataset=dataset, shuffle=False)  # batch_size=1
    img, _ = next(iter(dataloader))
    return img


def transform():
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transform


def get_labels():
    # Download class labels from imagenet dataset
    download_url("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json", ".",
                 "imagenet_class_index.json")

    with open("imagenet_class_index.json", "r") as h:
        labels = json.load(h)
    return labels
