import matplotlib.pyplot as plt
import numpy as np
import shap
from copy import deepcopy

def explain(model, img, org_img, labels):

    img = deepcopy(img)
    org_img = deepcopy(org_img)


    explainer = shap.GradientExplainer(model=model, data=img, local_smoothing=0.5)
    shap_values, indexes = explainer.shap_values(X=img, ranked_outputs=1, nsamples=200)

    labels = np.vectorize(lambda x: labels[str(x)][1])(indexes)
    shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]

    ##############Move to Shap folder#######################
    from matplotlib.colors import LinearSegmentedColormap
    cmap = plt.cm.bwr
    bwr = cmap(np.arange(cmap.N))
    bwr[:, -1] = np.linspace(0, 1, cmap.N)
    bwr = LinearSegmentedColormap.from_list("bwr", bwr)
    #######################################################

    shap.image_plot(shap_values=shap_values, pixel_values=org_img.reshape(1, 224, 224, 3)/255, cmap=bwr, labels=labels, show=False)

    fig = plt.gcf()
    plt.close()

    return fig
