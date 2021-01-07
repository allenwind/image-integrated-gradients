import glob
import numpy as np
from gradient_cam import get_gradcam_weights, visualize
from model_pretrain import model
from tensorflow.keras import preprocessing
from tensorflow.keras.applications.imagenet_utils import preprocess_input

size = (299, 299)
last_conv_layer_name = "block14_sepconv2_act"
classifier_layer_names = [
    "avg_pool",
    "dense", # prediction output
]

images = glob.glob("images/*.jpg")
for img_path in images:
    img = preprocessing.image.load_img(img_path, target_size=size)
    array = preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    img_array = preprocess_input(array, mode="tf")
    weights = get_gradcam_weights(
        model,
        img_array,
        last_conv_layer_name,
        classifier_layer_names
    )
    visualize(img, weights)
