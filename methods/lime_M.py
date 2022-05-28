from PIL import Image 
import matplotlib.pyplot as plt
import numpy as np
import cv2

from torch import double, float64

from torchvision import transforms
from lime import lime_image
from skimage.segmentation import mark_boundaries




# explanation 

def explain(model, img, file, model_str ):
    print("----------TEST-------------")

    # load picture
    org_img = np.array(cv2.imread("./data/images/" + file))
    org_img = np.asarray(cv2.resize(org_img, (224, 224), interpolation=cv2.INTER_CUBIC))
    org_img = org_img / 255.0

    # explainer
    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(img.reshape(-1, 224, 224, 3), model, top_labels=5, hide_color=0, num_samples=1000)

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))

    
    
    
    








