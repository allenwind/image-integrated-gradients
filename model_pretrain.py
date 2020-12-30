from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.applications import xception

import dataset

# 99.5% acc 1 epochs

train_dataset, valid_dataset = dataset.load_cats_vs_dogs(
    batch_size=32,
    image_shape=(299, 299)
)

pretrain_model = xception.Xception(weights="imagenet")
pretrain_model.trainable = False

pool = pretrain_model.layers[-2].output
output = Dense(2, activation="softmax")(pool)
model = Model(inputs=pretrain_model.input, outputs=output)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

if __name__ == "__main__":
    epochs = 1
    model.fit(train_dataset, epochs=epochs, validation_data=valid_dataset)
    model.save_weights("cat_vs_dog_weights")
else:
    model.load_weights("cat_vs_dog_weights")
