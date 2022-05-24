from pytorch_grad_cam import GradCAM
import cv2
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import numpy as np
import torchvision.transforms as transforms
import os


def explain(model, img, file, model_str):
    # loading original picture
    org_img = np.array(cv2.imread("./data/images/" + file))
    org_img = np.asarray(cv2.resize(org_img, (224, 224), interpolation=cv2.INTER_CUBIC))
    org_img = org_img / 255.0

    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)  # use_cuda=args.use_cuda
    grayscale_cam = cam(input_tensor=img, targets=None, aug_smooth=True, eigen_smooth=True)[0, :]
    visualization = show_cam_on_image(org_img, grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_JET)

    name = os.path.splitext(file)[0]
    name = name + "_" + model_str
    im = Image.fromarray(visualization)
    im.save("results/gradCam/" + name + ".jpg")
