from methods import data_handler, gradcam, LRP, SHAP, lime, integrated_gradients, confidence_scores
from models import AlexNet, Vgg16
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import torch
from datetime import datetime
import random



# define models
models_list = [Vgg16(), AlexNet()]
[model.eval() for model in models_list]

labels = data_handler.get_labels()
model_used = models_list[0]
for img_idx in range(4):

    img_org_np, img_prep_torch, img_name, img_true_label_str = data_handler.get_question_image(
            r'/Users/tobiaslabarta/Documents/GitHub/development/data/images',
            img_idx,
            labels)

    # for model in models_list:
    #     model_used = model
        
    # predict
    output = model_used.predict(img_prep_torch)
    # output has unnormalized scores. To get probabilities, run a softmax on it.
    pred_idx = torch.nn.functional.softmax(output[0], dim=0).detach().numpy().argmax()
    label = labels[str(pred_idx)]
    # must be manually verfied if true, because there are no true labels available for manually
    # downloaded images
    print(f"{img_name}, {model_used.name}: {label[1]}")


    # gradcam.explain(model_used.model, img_prep_torch, img_org_np).savefig(os.path.join("introduction" , f"intro_gradCAM_{model.name}_{img_name}"))
    # LRP.explain(model_used.model, img_prep_torch, img_name, model_used.name).savefig(os.path.join("introduction", f"intro_LRP_{model.name}_{img_name}"))
    # lime_ex = lime.LIMEExplainer(model_used)
    # lime_ex.explain(img_org_np).savefig(os.path.join("introduction", f"intro_LIME_{model.name}_{img_name}"))
    fig =SHAP.explain(model_used.model, img_prep_torch, img_org_np, labels).savefig(os.path.join("introduction", f"intro_SHAP_{model.name}_{img_name}"))
    print(fig)
    #ige = integrated_gradients.IntegratedGradientsExplainer(model_used)
    # ige.explain(img_prep_torch).savefig(os.path.join("introduction", f"intro_IntegratedGradients_{model.name}_{img_name}"))
    #confidence_scores.explain(model_used, img_prep_torch, labels, 3).savefig(os.path.join("introduction", f"intro_ConfidenceScores_{model.name}_{img_name}"))
