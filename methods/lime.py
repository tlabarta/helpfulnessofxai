from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from torchvision import transforms
import torch.nn.functional as F
from lime import lime_image
from skimage.segmentation import mark_boundaries
from methods import data_handler
import os
from models import AlexNet

# explanation

class LIMEExplainer():

    def __init__(self, model, ):
        self.model = model

    # def explain(self, img, file):
    #     # load picture
    #     org_img = np.array(cv2.imread("./data/images/" + file))
    #     org_img = np.asarray(cv2.resize(org_img, (224, 224), interpolation=cv2.INTER_CUBIC))
    #     org_img = org_img / 255.0

    #     explainer = lime_image.LimeImageExplainer()
    #     explanation = explainer.explain_instance(org_img.reshape(224, 224, 3), self.batch_predict, top_labels=5,
    #                                              hide_color=0, num_samples=1000)

    #     temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5,
    #                                                 hide_rest=True)

    #     filename = os.path.splitext(file)[0]
    #     filename = filename + "_" + self.model.name + ".png"
    #     output_path = "./results/LIME/" + filename

    #     plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    #     plt.savefig(output_path)


    def explain(self, img_org):
        org_img = np.array(cv2.imread("./data/images/" + file))
        org_img = np.asarray(cv2.resize(org_img, (224, 224), interpolation=cv2.INTER_CUBIC))
        org_img = org_img / 255.0
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(img_org.reshape(224, 224, 3), self.batch_predict, top_labels=5,
                                                 hide_color=0, num_samples=1000)

        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5,
                                                    hide_rest=True)

        plt.imshow(mark_boundaries(temp, mask)) # / 2 + 0.5

        # heatmap 
        ind =  explanation.top_labels[0]
        dict_heatmap = dict(explanation.local_exp[ind])
        heatmap = np.vectorize(dict_heatmap.get)(explanation.segments) 
        plt.imshow(heatmap, cmap = 'RdBu')
        plt.axis('off')
        fig = plt.gcf()
        plt.close()
        
        return fig 


    def batch_predict(self, imgs):
        """
        LIME needs a classifier_fn function that outputs probabilities. At the same time it also requires
        an image in numpy array shape. Thus the needed torch conversion needs to be done in this method.
        An extra transforming method (additionally to the one defined in data_handler) is needed because
        resizing is only possilbe on Pillow Images and not on numpy arrays.
        """
        transf = self.get_preprocess_transform()
        torch_imgs = torch.stack(tuple(transf(img) for img in imgs), dim=0).float()
        logits = self.model.predict(torch_imgs)
        probs = F.softmax(logits, dim=1)

        return probs.detach().numpy()

    def get_preprocess_transform(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transf = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        return transf

