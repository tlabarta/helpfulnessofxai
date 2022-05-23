from pytorch_grad_cam import GradCAM
import cv2
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import numpy as np
import torchvision.transforms as transforms


def explain(model, imgpath):
    img = np.asarray(cv2.imread(imgpath))
    img = np.squeeze(img)
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = np.float32(img) / 255
    input_tensor = preprocess_image(img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    target_layers = [model.features[-1]]

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)  # use_cuda=args.use_cuda
    grayscale_cam = cam(input_tensor=input_tensor, targets=None, aug_smooth=True, eigen_smooth=True)[0, :]
    visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_JET)  #
    im = Image.fromarray(visualization)
    im.save("results/gradcam/gradcam.jpg")
