from methods import alexnet_model, vgg_model, data_handler, gradcam

# define model
model = vgg_model.train()

# import image
img_folder = './data/'
img = data_handler.get_image(img_folder)

# let model do a prediction
predictions = vgg_model.predict(model, img)

# get all class labels
labels = data_handler.get_labels()

# use xai to explain model prediction
imgpath = './data/images/gazelle.jpg'
gradcam.explain(model, imgpath)
