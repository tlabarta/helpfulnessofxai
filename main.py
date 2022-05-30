from methods import data_handler, gradcam, LRP, SHAP, lime,contrastive_explanation
import models
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt


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


    for i in range(args.num_images):
        img, _ = next(data)

        org_img = np.array(cv2.imread(args.img_folder+"images/"+files[i]))
        org_img = np.asarray(cv2.resize(org_img, (224, 224), interpolation=cv2.INTER_CUBIC), dtype=np.float32)
        org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

        for model in models_list:

            LRP.explain(model.model,img, files[i], model.name)
            gradcam.explain(model.model,img,org_img,files[i],model.name)
            SHAP.explain(model.model, img, org_img, files[i], labels, model.name)

            model_dict = dict(type=model.name, arch=model.model, layer_name=model.ce_layer_name, input_size=(224, 224))
            ce = contrastive_explanation.ContrastiveExplainer(model_dict)
            # # Choice of contrast; The Q in `Why P, rather than Q?'. Class 130 is flamingo
            ce.explain(org_img, img, 130, f"./results/ContrastiveExplanation/{model.name}_{files[i]}")


            #TODO Adjust this a bit so we dont initilize the model eight times only two
            lime_ex = lime.LIMEExplainer(model)
            lime_ex.explain(img, files[i])

if __name__ == '__main__' :
    main()
