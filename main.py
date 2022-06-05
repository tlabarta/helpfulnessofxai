from methods import data_handler, gradcam, LRP, SHAP, lime, integrated_gradients
import models
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import torch


# TODO gradcam

def main():
    parser = argparse.ArgumentParser(description='run explain methods')
    parser.add_argument('--VGG', type=bool, default=True)
    parser.add_argument('--AlexNet', type=bool, default=True)
    parser.add_argument('--LRP', type=bool, default=False)
    parser.add_argument('--gradCam', type=bool, default=False)
    parser.add_argument('--Lime', type=bool, default=False)
    parser.add_argument('--CEM', type=bool, default=False)
    parser.add_argument('--SHAP', type=bool, default=False)
    parser.add_argument('--num_images', type=int, default=4)
    parser.add_argument('--img_folder', type=str, default='./data/')
    args = parser.parse_args()


    # define models
    models_list = []
    if args.VGG :
        vgg = models.Vgg16()
        models_list.append(vgg)
    if args.AlexNet :
        alex = models.AlexNet()
        models_list.append(alex)

    # import image
    data = data_handler.get_image(args.img_folder)
    files = data_handler.get_files(args.img_folder)
    files.sort()
    labels = data_handler.get_labels()

    # for i in range(args.num_images):
    #     img, _ = next(data)

    #     org_img = np.array(cv2.imread(args.img_folder+"images/"+files[i]))
    #     org_img = np.asarray(cv2.resize(org_img, (224, 224), interpolation=cv2.INTER_CUBIC), dtype=np.float32)
    #     org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

    #     for model in models_list:

    #         LRP.explain(model.model,img, files[i], model.name)
    #         gradcam.explain(model.model,img,org_img,files[i],model.name)
    #         SHAP.explain(model.model, img, org_img, files[i], labels, model.name)

    #         model_dict = dict(type=model.name, arch=model.model, layer_name=model.ce_layer_name, input_size=(224, 224))
    #         ce = contrastive_explanation.ContrastiveExplainer(model_dict)
    #         # # Choice of contrast; The Q in `Why P, rather than Q?'. Class 130 is flamingo
    #         ce.explain(org_img, img, 130, f"./results/ContrastiveExplanation/{model.name}_{files[i]}")


    #         #TODO Adjust this a bit so we dont initilize the model eight times only two
    #         lime_ex = lime.LIMEExplainer(model)
    #         lime_ex.explain(img, files[i])
    
    # load questionaire_list from .json or .pickle
    questionaires_list = data_handler.get_questionaires("data2/questionaires.pickle")
    if not os.path.exists("/questionaire_forms/"):
            os.mkdir("/questionaire_forms/")
    
    for idx, questionaire in enumerate(questionaires_list):
        folder_path = f"/questionaire_forms/questionaire_{idx+1}"
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        # create questionaire_n subfolders
        for qu_idx, question in enumerate(questionaire):
            print(question)
            # (7127, 'vgg', 'LIME', False)
            # easiest way: xai-methods should return their figure; then save here into appropriate folder

            # load image by index
            img_idx, model_name_used, xai_used, bool_used  = question
            model_used = models.Vgg16() if model_name_used == "vgg" else models.AlexNet()
            model_used.train()
            img_org_np, img_prep_torch, img_name = data_handler.get_question_image(r'C:\Users\julia\Dokumente\GitHub\development\data2\imagenetv2-matched-frequency-format-val', img_idx)
            
            # only for testing purposes
            # print(img_prep_torch)
            # output = model_used(img_prep_torch)
            # probabilities = torch.nn.functional.softmax(output[0], dim=0)
            # print(probabilities.argmax(), probabilities.max())
            # print(img_name)


            if xai_used == "gradCAM":
                fig_explanation = gradcam.explain(model_used.model,img_prep_torch, img_org_np, img_name, model_used.name)
                print(fig_explanation)
            elif xai_used == "LRP":
                fig_explanation = LRP.explain(model_used.model,img_prep_torch, img_name, model_used.name)
                print(fig_explanation)
            elif xai_used == "LIME":
                lime_ex = lime.LIMEExplainer(model_used)
                fig_explanation = lime_ex.explain(img_org_np)
            elif xai_used == "SHAP":
                fig_explanation = SHAP.explain(model_used.model, img_prep_torch, img_org_np, img_name, labels, model_used.name)
            elif xai_used == "IntegratedGradients":
                ige = integrated_gradients.IntegratedGradientsExplainer(model_used)
                fig_explanation = ige.explain(img_prep_torch, img_name)
            elif xai_used == "ConfidenceScores":
                pass
            
            # save explanation for current question in appropriate questionaire folder
            fig_explanation.savefig(f"{folder_path}/{qu_idx+1}_{model_name_used}_{bool_used}_{xai_used}_{img_name}")
            

if __name__ == '__main__' :
    main()
