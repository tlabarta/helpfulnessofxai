import os
import matplotlib.pyplot as plt
import numpy as np
import shap


def explain(model, img, org_img, files, labels, model_str):

    name = os.path.splitext(files)[0]
    name = name + "_" + model_str

    explainer = shap.GradientExplainer(model=model, data=img, local_smoothing=0.5)
    shap_values, indexes = explainer.shap_values(X=img, ranked_outputs=1, nsamples=200)

    labels = np.vectorize(lambda x: labels[str(x)][1])(indexes)
    shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]


    shap.image_plot(shap_values=shap_values, pixel_values=org_img.reshape(1, 224, 224, 3)/255, labels=labels, show=False)
    plt.savefig("results/shap/" + name + ".jpg")