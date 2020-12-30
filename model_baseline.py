import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.layers import *
import dataset

train_dataset, valid_dataset = dataset.load_cats_vs_dogs(batch_size=32)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(256, 256, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

if __name__ == "__main__":
    epochs = 10
    model.fit(train_dataset, epochs=epochs, validation_data=valid_dataset)
    model.save_weights("cat_vs_dog_baseline_weights")
else:
    model.load_weights("cat_vs_dog_baseline_weights")
