import torch, torchvision
from torch import nn
from torchvision import transforms, models, datasets
from PIL import Image
import shap
import numpy as np
import data_handler



def explain(model, imgpath):
    img = data_handler.get_image(imgpath)
    dataset = data_handler.

    explainer = shap.GradientExplainer(model=model, data=img, local_smoothing=0.5)
    shap_values, indexes = explainer.shap_values(img, ranked_outputs=1, nsamples=200)

    index_names = np.vectorize(lambda x: class_names[str(x)][1])(indexes)
    visualization = shap.image_plot(shap_values=shap_values, pixel_values=img, labels=index_names)
    im = Image.fromarray(visualization)
    im.save("results/shap/shap.jpg")