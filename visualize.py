import glob
from integrated_gradients import visualize
from model_pretrain import model
from tensorflow.keras import preprocessing

images = glob.glob("images/*.jpg")
for img_path in images:
    img = preprocessing.image.load_img(img_path, target_size=(299, 299))
    array = preprocessing.image.img_to_array(img)
    visualize(model, array, label=None)
