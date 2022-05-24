import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
from PIL import Image
from methods import data_handler
import os


def explain(model, img, files, labels, model_str):

    name = os.path.splitext(files)[0]
    name = name + "_" + model_str

    print("SHAP started")
    explainer = shap.GradientExplainer(model=model, data=img, local_smoothing=0.5)
    shap_values, indexes = explainer.shap_values(X=img, ranked_outputs=1, nsamples=200)


    labels = np.vectorize(lambda x: labels[str(x)][1])(indexes)
    shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]

    print("SHAP finished")

    shap.image_plot(shap_values=shap_values, pixel_values=img.numpy().reshape(-1,224,224,3), labels=labels, show=False)
    plt.savefig("results/shap/" + name + ".jpg")