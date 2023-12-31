import itertools
import os
import pandas as pd
import pickle
import random
import torch
from torchvision import datasets
from tqdm import tqdm
from scipy import rand

import data_handler
from models import AlexNet, Vgg16


def generate_model_testset_results(model, testset_path):
    """
    Evaluate whole 'imagenetv2-matched-frequency-format-val' dataset
    on given model and saves the results ("img_name", "max_confidence", "pred_label", "true_label" for each image) 
    in a DataFrame via pickle.
    
    For information on the dataset see https://github.com/modestyachts/ImageNetV2
    """
    img_folder = datasets.ImageFolder(root=testset_path)

    img_names, true_labels_idx, pred_labels_idx, pred_max_confidences = [], [], [], []

    for img_path in tqdm(img_folder.imgs):
        pil_img = img_folder.loader(img_path[0])
        img_name = img_path[0].split(os.sep)[-1]

        # preprocessing and prediction
        input_tensor = data_handler.transform()(pil_img)
        input_tensor = input_tensor.unsqueeze(0)
        output = model.predict(input_tensor)
        # output has unnormalized scores. To get probabilities, run a softmax on it.
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        img_names.append(img_name)
        pred_max_confidences.append(probabilities.detach().numpy().max())
        pred_labels_idx.append(probabilities.detach().numpy().argmax())
        true_labels_idx.append(int(img_path[0].split(os.sep)[-2]))

    df = pd.DataFrame([img_names, pred_max_confidences, pred_labels_idx, true_labels_idx]).transpose()
    df.columns = ["img_name", "max_confidence", "pred_label", "true_label"]
    df["pred_is_correct"] = df["pred_label"] == df["true_label"]

    return df


def create_questionnairs(imgs_idx, xai_methods, model_names, df_vgg, df_alex, seed=None):
    if seed:
        random.seed(seed)
    # create first half of question with fixed images for all questionnaire forms
    questionnaires_list = get_fixed_img_questionnaires(imgs_idx, xai_methods, model_names)
    # adding images works directly on the reference of 'questionnaires_list'
    add_random_unique_images(questionnaires_list, imgs_idx, df_alex, df_vgg, model_names, xai_methods)

    return questionnaires_list


def get_fixed_img_questionnaires(imgs_idx, xai_methods, models):
    NUM_QUESTIONNAIRES = 12
    NUM_IMGS = 12
    questionnaires_list = []
    random_imgs_idx = [imgs_idx.pop(random.randint(0, len(imgs_idx) - 1)) for i in range(NUM_IMGS)]
    permutations = list(itertools.product(random_imgs_idx, models, xai_methods))
    # distribute permutations on questionnaires 
    for q in range(NUM_QUESTIONNAIRES):
        questionnaire = []
        for i in range(NUM_IMGS):
            if (q + i) > (NUM_IMGS - 1):
                questionnaire.append(permutations[i * NUM_IMGS:i * NUM_IMGS + NUM_IMGS][(q + i) - NUM_IMGS])
            else:
                questionnaire.append(permutations[i * NUM_IMGS:i * NUM_IMGS + NUM_IMGS][q + i])
        questionnaires_list.append(questionnaire)

    return questionnaires_list


def add_random_unique_images(questionnaires_list, imgs_idx, df_alex, df_vgg, model_names, xai_methods):
    FINAL_QUESTIONNAIRE_SIZE = 24

    for idx_qn, questionnaire in enumerate(questionnaires_list):

        df_variants_count = pd.DataFrame(list(itertools.product(xai_methods, model_names, [True, False]))).groupby(
            [0, 1, 2]).count()
        df_variants_count["count"] = 0

        # evaluate variants for the already drawn fixed questions
        for idx_q, question in enumerate(questionnaire):
            if question[1] == "alex":
                if df_alex["pred_is_correct"][question[0]]:
                    questionnaires_list[idx_qn][idx_q] += (True,)
                    df_variants_count.loc[question[2], "alex", True]["count"] += 1
                else:
                    questionnaires_list[idx_qn][idx_q] += (False,)
                    df_variants_count.loc[question[2], "alex", False]["count"] += 1
            else:
                if df_vgg["pred_is_correct"][question[0]]:
                    questionnaires_list[idx_qn][idx_q] += (True,)
                    df_variants_count.loc[question[2], "vgg", True]["count"] += 1
                else:
                    questionnaires_list[idx_qn][idx_q] += (False,)
                    df_variants_count.loc[question[2], "vgg", False]["count"] += 1
        """
        add addtional random images to each questionnaire such that for every variant in df_variants_count the 
        count will be 1
        """
        while df_variants_count["count"].sum() != FINAL_QUESTIONNAIRE_SIZE:
            rand_img_idx = imgs_idx.pop(random.randint(0, len(imgs_idx) - 1))

            alex_pred = df_alex.loc[rand_img_idx]["pred_is_correct"]
            vgg_pred = df_alex.loc[rand_img_idx]["pred_is_correct"]

            df_alex_options = df_variants_count.loc[:, "alex", alex_pred]
            df_alex_options = df_alex_options[df_alex_options["count"] == 0]

            df_vgg_options = df_variants_count.loc[:, "vgg", vgg_pred]
            df_vgg_options = df_vgg_options[df_vgg_options["count"] == 0]

            if not df_alex_options.empty:
                rand_variant = df_alex_options.index[random.randint(0, df_alex_options.shape[0] - 1)]
                question = (rand_img_idx, rand_variant[1], rand_variant[0], rand_variant[2])
                questionnaire.append(question)
                df_variants_count.loc[rand_variant]["count"] += 1

            elif not df_vgg_options.empty:
                rand_variant = df_vgg_options.index[random.randint(0, df_vgg_options.shape[0] - 1)]
                question = (rand_img_idx, rand_variant[1], rand_variant[0], rand_variant[2])
                questionnaire.append(question)
                df_variants_count.loc[rand_variant]["count"] += 1


def save_questionnaires(questionnaires_list, path):
    with open(path, 'wb') as f:
        pickle.dump(questionnaires_list, f)


def shuffle_questions(questionnaire):
    for questionnaire in questionnaire:
        random.shuffle(questionnaire)


def main():
    """
    create questionnaires
    must only be evaluated if testset hasn't already been evaluated
    """
    folder_vgg = os.path.join(os.path.curdir, "data", "stats", "df_vgg.pickle")
    folder_alex = os.path.join(os.path.curdir, "data", "stats", "df_alexnet.pickle")
    if not (os.path.exists(folder_alex) and os.path.exists(folder_vgg)):
        models = [Vgg16(), AlexNet()]

        for model in models:
            model.eval()
            folder = os.path.join(os.path.curdir, "data", "imagenetv2-matched-frequency-format-val")
            df = generate_model_testset_results(model, folder)
            df.to_pickle(f"data/stats/df_{model.name}_2.pickle")

        imgs_idx = list(range(10000))
        xai_methods = ['gradCAM', 'LRP', 'SHAP', 'LIME', 'ConfidenceScores', 'IntegratedGradients']
        model_names = ["alex", "vgg"]

        df_vgg = pd.read_pickle(folder_vgg)
        df_alex = pd.read_pickle(folder_alex)

        questionnaires_list = create_questionnairs(imgs_idx, xai_methods, model_names, df_vgg, df_alex, seed=3)
        shuffle_questions(questionnaires_list)
        folder = os.path.join(os.path.curdir, "data", "question_generation", "questionnaires_shuffled.pickle")
        save_questionnaires(questionnaires_list, folder)

    # additionally shuffle questions in questionnairs
    folder = os.path.join(os.path.curdir, "data", "question_generation", "questionnaires.pickle")
    questionnaires_list = data_handler.get_questionnaires(folder)

    shuffle_questions(questionnaires_list)

    folder = os.path.join(os.path.curdir, "data", "question_generation", "questionnaires_shuffled.pickle")
    save_questionnaires(questionnaires_list, folder)


if __name__ == '__main__':
    main()
