from methods import data_handler, gradcam, LRP
import models
import argparse
import numpy as np
import cv2


# TODO use argparse for single mpdules once all explain methods are in here
# maybe noch confidence scores als arg hinzufügen
def main():
    parser = argparse.ArgumentParser(description='run explain methods')
    parser.add_argument('--VGG', type=bool, default=True)
    parser.add_argument('--AlexNet', type=bool, default=False)
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
    if args.VGG:
        vgg = models.Vgg16()
        models_list.append(vgg)
    if args.AlexNet:
        alex = models.AlexNet()
        models_list.append(alex)

    # import image
    data = data_handler.get_image(args.img_folder)
    files = data_handler.get_files(args.img_folder)
    labels = data_handler.get_labels()

    for i in range(args.num_images):
        img, _ = next(data)

        for model in models_list:
            LRP.explain(model.model, img, files[i], model.name)
            gradcam.explain(model.model, img, files[i], model.name)

    #preds = vgg.predict(img)
    #print(data_handler.topk_confidence_scores(preds, labels, 5))


if __name__ == '__main__':
    main()
