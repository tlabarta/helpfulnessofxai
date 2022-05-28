from PIL import Image 
import matplotlib.pyplot as plt
import numpy as np
import cv2

from torch import double, float64

from torchvision import transforms
from lime import lime_image
from skimage.segmentation import mark_boundaries

# explanation 

def lime_method(img, model):
    print("----------TEST-------------")
    print(np.array(img))
    explainer = lime_image.LimeImageExplainer()
    # explanation = explainer.explain_instance(np.array(img).astype(float64).reshape(224, 224, 3) , 
    # img.numpy().reshape(224, 224, 3).astype("double")
    explanation = explainer.explain_instance(img.reshape(224, 224, 3).to(double),
                                                model, # classification function
                                                top_labels=5, 
                                                hide_color=0, 
                                                num_samples=1000) 
    
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False) 
    img_boundry1 = mark_boundaries(temp/255.0, mask)
    plt.imshow(img_boundry1)
    # pic.save("results/lime/lime_test.jpg")
                                   





