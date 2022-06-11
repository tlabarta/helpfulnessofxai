from torchvision import datasets
from torch.utils import data
import torchvision.transforms as transforms
import json
from torchvision.datasets.utils import download_url
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt



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
    files = os.listdir(path+images)

    return files


def get_labels():
    # Download class labels from imagenet dataset
    download_url("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json", ".",
                 "data/imagenet_class_index.json")

    with open("data/imagenet_class_index.json", "r") as h:
        labels = json.load(h)
        
    return labels


def get_question_image(testset_path, img_idx, labels):
    img_folder = datasets.ImageFolder(root=testset_path)
    img_path = img_folder.imgs[img_idx][0]
    pil_img = img_folder.loader(img_path)
    img_org_np = np.expand_dims(np.array(pil_img.resize((224, 224))), 0)
    img_name = img_path.split("/")[-1]
    #img_name = img_path.split("\\")[-1]
    print(img_path)
    # preprocessing
    img_prep_torch = transform()(pil_img)
    img_prep_torch = img_prep_torch.unsqueeze(0)
    # extract correct class
    class_idx_true_str = img_path.split("/")[-2]
    #class_idx_true_str = img_path.split("\\")[-2]
    img_label_true = labels[class_idx_true_str][1]

    return img_org_np, img_prep_torch, img_name, img_label_true


def get_questionaires(path):
    with open(path,'rb') as f:
        questionaires_list = pickle.load(f)
    
    return questionaires_list


def get_figure_from_img_array(image_np, title):
    plt.imshow(image_np)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    fig = plt.gcf()
    plt.close()

    return fig
            
