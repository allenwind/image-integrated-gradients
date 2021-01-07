import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices("GPU")
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras import preprocessing
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm

# gradient class activation heatmap

def get_gradcam_weights(
    model,
    image,
    last_conv_layer_name,
    classifier_layer_names,
    normalize=True):
    conv_layer = model.get_layer(last_conv_layer_name)
    conv_model = tf.keras.Model(model.inputs, conv_layer.output)

    classifier_input = tf.keras.Input(shape=conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier = tf.keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        conv_layer_output = conv_model(image)
        tape.watch(conv_layer_output)
        preds = classifier(conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    grads = tape.gradient(top_class_channel, conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_layer_output = conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        conv_layer_output[:, :, i] *= pooled_grads[i]

    weights = np.mean(conv_layer_output, axis=-1)
    if normalize:
        weights = np.maximum(weights, 0) / np.max(weights)
    return weights

def convert_to_heatmap(weights, shape, cm="jet"):
    weights = np.uint8(255 * weights)
    cmap = matplotlib.cm.get_cmap(cm)
    colors = cmap(np.arange(256))[:, :3]
    heatmap = colors[weights]
    heatmap = preprocessing.image.array_to_img(heatmap)
    heatmap = heatmap.resize(shape)
    heatmap = preprocessing.image.img_to_array(heatmap)
    return heatmap

def superimpose_image(image1, image2, alpha=0.6):
    superimposed_image = image1 + image2 * alpha
    return preprocessing.image.array_to_img(superimposed_image)

def visualize(image, weights):
    _, ax = plt.subplots(1, 3, figsize=(15, 8))
    ax[0].imshow(image)
    ax[0].set_title("image")

    ax[1].matshow(weights)
    ax[1].set_title("heatmap")

    image_array = preprocessing.image.img_to_array(image)
    heatmap = convert_to_heatmap(weights, image_array.shape[:-1])
    superimposed_img = superimpose_image(image_array, heatmap)
    ax[2].imshow(superimposed_img)
    ax[2].set_title("superimposed image")
    plt.show()
