import argparse
import os
from datetime import datetime

import data_handler
import models
from methods import gradcam, LRP, SHAP, lime, integrated_gradients, confidence_scores


def main():
    parser = argparse.ArgumentParser(description='run explain methods')
    parser.add_argument('--LRP', type=bool, default=True)
    parser.add_argument('--gradCam', type=bool, default=True)
    parser.add_argument('--LIME', type=bool, default=True)
    parser.add_argument('--IntGrad', type=bool, default=True)
    parser.add_argument('--CS', type=bool, default=True)
    parser.add_argument('--SHAP', type=bool, default=True)
    args = parser.parse_args()


    labels = data_handler.get_labels()

    # load questionnaire_list from .json or .pickle
    folder = os.path.join(os.path.curdir, "data","question_generation","questionnaires.pickle")
    questionnaires_list = data_handler.get_questionnaires(folder)

    # create root folder for questionnaires
    current_dir = os.getcwd()
    folder_path = os.path.join(current_dir, f"questionnaire_forms_{datetime.now().strftime('%d-%m_%H-%M')}")
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    # create all questionnaires according to questionnares_list
    for idx, questionnaire in enumerate(questionnaires_list):
        sub_folder_path = os.path.join(folder_path, f"questionnaire_{idx + 1}")
        if not os.path.exists(sub_folder_path):
            os.mkdir(sub_folder_path)

        # create questionnaire_n subfolders
        for qu_idx, question in enumerate(questionnaire):

            # load image by index
            img_idx, model_name_used, xai_used, bool_used = question

            model_used = models.Vgg16() if model_name_used == "vgg" else models.AlexNet()
            model_used.eval()

            folder = os.path.join(os.path.curdir, "data", "imagenetv2-matched-frequency-format-val")
            img_org_np, img_prep_torch, img_name, img_true_label_str = data_handler.get_question_image(folder, img_idx,
                                                                                                       labels)
            fig_explanation = None
            if xai_used == "gradCAM" and args.gradCam:
                fig_explanation = gradcam.explain(model_used.model, img_prep_torch, img_org_np)

            elif xai_used == "LRP" and args.LRP:
                fig_explanation = LRP.explain(model_used.model, img_prep_torch, img_name, model_used.name)

            elif xai_used == "LIME" and args.LIME:
                lime_ex = lime.LIMEExplainer(model_used)
                fig_explanation = lime_ex.explain(img_org_np)

            elif xai_used == "SHAP" and args.SHAP:
                fig_explanation = SHAP.explain(model_used.model, img_prep_torch, img_org_np, labels)

            elif xai_used == "IntegratedGradients" and args.IntGrad:
                ige = integrated_gradients.IntegratedGradientsExplainer(model_used)
                fig_explanation = ige.explain(img_prep_torch)

            elif xai_used == "ConfidenceScores" and args.CS:
                fig_explanation = confidence_scores.explain(model_used, img_prep_torch, labels, 3)

            # save explanation and original image for current question in appropriate questionnaire folder
            if fig_explanation:
                fig_explanation.savefig(
                    os.path.join(sub_folder_path, f"{qu_idx + 1}_{model_name_used}_{bool_used}_{xai_used}_{img_name}"))

                fig_org = data_handler.get_figure_from_img_array(img_org_np[0], f"True class: {img_true_label_str}")
                fig_org.savefig(os.path.join(sub_folder_path, f"{qu_idx + 1}_org_{img_name}"))


if __name__ == '__main__':
    main()
