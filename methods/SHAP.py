import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
from PIL import Image
import os
import torchvision.transforms as transforms


def explain(model, img, org_img, files, labels, model_str):

    name = os.path.splitext(files)[0]
    name = name + "_" + model_str

    print("SHAP started")
    explainer = shap.GradientExplainer(model=model, data=img, local_smoothing=0.5)
    shap_values, indexes = explainer.shap_values(X=img, ranked_outputs=1, nsamples=200)

    print(type(shap_values))
    print(shap_values)



    labels = np.vectorize(lambda x: labels[str(x)][1])(indexes)
    shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]


    print("SHAP finished")
    #img.numpy().reshape(-1, 224, 224, 3)

    print(org_img.dtype)
    print(org_img.shape)
    print(org_img.reshape(1, 224, 224, 3))

    shap.image_plot(shap_values=shap_values, pixel_values=org_img.reshape(1, 224, 224, 3)/255, labels=labels, show=False)
    plt.savefig("results/shap/" + name + ".jpg")