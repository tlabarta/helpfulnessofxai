import cmapy
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import torchvision.transforms as transforms
from copy import deepcopy
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image


def explain(model, img, org_img):
    img = deepcopy(img)
    org_img = deepcopy(org_img)

    org_img = np.float32(org_img) / 255
    org_img = np.matmul(org_img[..., :3], [0.299, 0.587, 0.114]).squeeze()
    org_img = org_img[:, :, np.newaxis]
    target_layers = [model.features[-1]]

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)  # use_cuda=args.use_cuda
    grayscale_cam = cam(input_tensor=img, targets=None, aug_smooth=True, eigen_smooth=True)[0, :]
    visualization = show_cam_on_image(org_img, grayscale_cam, use_rgb=True, colormap=cmapy.cmap("Reds"))
    plt.imshow(visualization)
    #plt.show()
    plt.axis('off')
    fig = plt.gcf()
    plt.close()

    return fig

