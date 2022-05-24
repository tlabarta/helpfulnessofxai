from methods import data_handler, gradcam, LRP, SHAP
import models
import matplotlib

# define model
vgg = models.Vgg16()
alex = models.AlexNet()

vgg.train()
alex.train()

matplotlib.use('TkAgg')

# import image
img_folder = './data/'
img = data_handler.get_image(img_folder)

# let model do a prediction
predictions = vgg.predict(img)

# get all class labels
labels = data_handler.get_labels()

# use xai to explain model prediction
imgpath = './data/images/gazelle.jpg'
#gradcam.explain(vgg.model, imgpath)

# Example for LRP
#LRP.LRP(imgpath,vgg.model,vgg.name)
print("Shap is started")
SHAP.explain(vgg.model, img, labels, vgg.name)
SHAP.explain(alex.model, img, labels, alex.name)
print("Shap is terminated")
