import alexnet_model
from vgg_model import predict
import data_handler
import gradcam

# define model
model = alexnet_model.train()

# import image
img_folder = './data/'
img = data_handler.get_image(img_folder)

# let model do a prediction
predictions = predict(model, img)

# get all class labels
labels = data_handler.get_labels()

# use xai to explain model prediction
imgpath = './data/images/gazelle.jpg'
gradcam.explain(model, imgpath)
