from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import keras.backend as K
import numpy as np
import json
import shap
import random

import tensorflow.compat.v1.keras.backend as Kf
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


random_pics = random.sample(range(0, 50), 2)
# load pre-trained model and choose five images to explain
model = VGG16(weights='imagenet', include_top=True)
X,y = shap.datasets.imagenet50()
to_explain = X[random_pics]
# load the ImageNet class names
url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
fname = shap.datasets.cache(url)
with open(fname) as f:
    class_names = json.load(f)
#run Gradient Explainer on Model
explainer = shap.GradientExplainer(model, preprocess_input(X.copy()),local_smoothing=0.5)
shap_values,indexes = explainer.shap_values(to_explain, ranked_outputs=1,nsamples=200)
# get the names for the classes
index_names = np.vectorize(lambda x: class_names[str(x)][1])(indexes)
# plot the explanations
shap.image_plot(shap_values=shap_values, pixel_values=to_explain, labels=index_names)