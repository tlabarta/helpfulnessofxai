from methods import data_handler, gradcam, LRP
import models

# TODO Argparse for dorect calling
# TODO nameming the output
# TODO gradcam

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
    files = data_handler.get_files(img_folder)
    labels = data_handler.get_labels()


    for i in range(num_images):
        img, _ = next(data)
        if Vgg :
            LRP.explain(img, files[i], vgg.model, vgg.name)
        if AlexNet :
            LRP.explain(img, files[i], alex.model, alex.name)

    """
    # load original: as long as nobody needs to use shuffle
    img = np.asarray(cv2.imread("./data/images/"+files[0]))
    img = np.asarray(cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC))
    """
if __name__ == '__main__' :
    main(True,False,4,'./data/')