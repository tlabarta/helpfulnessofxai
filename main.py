from methods import contrastive_explanation, data_handler, gradcam, LRP
import models

# define model
vgg = models.Vgg16()
alex = models.AlexNet()

vgg.train()
alex.train()

# import image
img_folder = './data/'
img = data_handler.get_image(img_folder)

print(img.shape)

# let model do a prediction
predictions = vgg.predict(img)

# get all class labels
labels = data_handler.get_labels()

# use xai to explain model prediction
imgpath = './data/images/gazelle.jpg'

# example for contrastive explanation
# print("----------------CE_VGG_START--------------------")
# vgg_model_dict = dict(type=vgg.name, arch=vgg.model, layer_name='features_29', input_size=(224, 224))
# ce = contrastive_explanation.ContrastiveExplainer(vgg_model_dict)
# ce.explain(img, img, 30, "./results/ContrastiveExplanation/gazelle.jpg")
# print("----------------CE_VGG_FINISHED--------------------")

print("----------------CE_ALEX_START--------------------")
alexnet_model_dict = dict(type=alex.name, arch=alex.model, layer_name='features_11', input_size=(224, 224))
ce = contrastive_explanation.ContrastiveExplainer(alexnet_model_dict)
ce.explain(img, img, 30, f"./results/ContrastiveExplanation/gazelle_{alex.name}.jpg")
print("----------------CE_ALEX_FINISHED--------------------")


# example for GradCAM
# gradcam.explain(vgg.model, imgpath)

# # example for LRP
# LRP.LRP(imgpath,vgg.model,vgg.name)




