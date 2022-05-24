from methods import data_handler, gradcam, LRP
import models

# TODO Argparse for dorect calling
def main(Vgg:bool,
         AlexNet : bool,
         num_images : int,
         img_folder: str,
         ):

    # define model
    if Vgg : vgg = models.Vgg16()
    if AlexNet : alex = models.AlexNet()

    # import image
    data = data_handler.get_image(img_folder)
    labels = data_handler.get_labels()

    imgpath = './data/images/gazelle.jpg'

    for i in range(num_images):
        img, _ = next(data)
        print(img.shape)
        print(img)
        if vgg :
            gradcam.explain(vgg.model, imgpath)
            LRP.LRP(img, imgpath, vgg.model, vgg.name)
        if alex :
            gradcam.explain(alex.model, imgpath)
            LRP.LRP(img, imgpath, alex.model, alex.name)



if __name__ == '__main__' :
    main(True,False,4,'./data/')