from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from lime import lime_image
from copy import deepcopy
from development.data_handler import transform

# explanation

class LIMEExplainer():

    def __init__(self, model, ):
        self.model = model

   

    def explain(self, img_org):

        img_org = deepcopy(img_org)

        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(img_org.reshape(224, 224, 3), self.batch_predict, top_labels=5, hide_color=0, num_samples=1000)

        img = Image.fromarray(img_org[0]).convert('L')

        #temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=5, hide_rest=True)
        # open the original image and grayscale it 

        # tmake image transparent
        img.putalpha(50)

        # resize image 
        new_width = 224
        new_height = 224
        img = img.resize((new_width, new_height), Image.ANTIALIAS)

        # heatmap
        ind =  explanation.top_labels[0]
        dict_heatmap = dict(explanation.local_exp[ind], positive_only = False)
        heatmap = np.vectorize(dict_heatmap.get)(explanation.segments) 

        fig , axes = plt.subplots()
        axes.imshow(heatmap, cmap = 'bwr',  vmin  = -heatmap.max(), vmax = heatmap.max())
        axes.imshow(img, cmap = 'gray', vmin = 0, vmax = 255)

        # plt.imshow(heatmap, cmap = 'bwr', vmin  = -heatmap.max(), vmax = heatmap.max())
  #      plt.colorbar() -> RuntimeError: No mappable was found to use for colorbar creation. First define a mappable such as an image (with imshow) or a contour set (with contourf).
        plt.axis("off")
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
        torch_imgs = torch.stack(tuple(transform()(img) for img in imgs), dim=0).float()
        logits = self.model.predict(torch_imgs)
        probs = F.softmax(logits, dim=1)

        return probs.detach().numpy()
