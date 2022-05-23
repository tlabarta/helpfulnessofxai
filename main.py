from methods import data_handler, gradcam, LRP, SHAP
import models
# define model
vgg = models.Vgg16()
alex = models.AlexNet()

vgg.train()
alex.train()

# import image
img_folder = './data/'
img = data_handler.get_image(img_folder)

# let model do a prediction
predictions = vgg.predict(img)

# get all class labels
labels = data_handler.get_labels()

# use xai to explain model prediction
imgpath = './data/'
#gradcam.explain(vgg.model, imgpath)

# Example for LRP
#LRP.LRP(imgpath,vgg.model,vgg.name)

SHAP.explain(vgg.model, imgpath)
