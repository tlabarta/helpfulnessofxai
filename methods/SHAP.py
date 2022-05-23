import numpy as np
import shap
import torch
from PIL import Image
from methods import data_handler


def explain(model, imgpath):
    img = data_handler.get_image(imgpath)
    index_names = data_handler.get_labels()

    img_transformed = data_handler.normalize(img)

    explainer = shap.GradientExplainer(model=model, data=img_transformed, local_smoothing=0.5)
    shap_values, indexes = explainer.shap_values(X=img_transformed, ranked_outputs=1, nsamples=200)

    index_names = np.vectorize(lambda x: index_names[str(x)][1])(indexes)
    shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]

    visualization = shap.image_plot(shap_values=shap_values, pixel_values=img, labels=index_names, show=True)

    im = Image.fromarray(visualization)
    im.save("results/shap/shap.jpg")