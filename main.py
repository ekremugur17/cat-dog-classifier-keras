# DOWNLOAD DATASET FROM UDACITY AND UNZIP THE FILE YOU DOWNLOADED
# !wget https://s3.amazonaws.com/content.udacity-data.com/nd089/Cat_Dog_data.zip
# !unzip -q Cat_Dog_data.zip -d dataset

import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

def plot_images(image_array):
  fig, axes = plt.subplots(1, 5, figsize=(20, 20))
  axes = axes.flatten()
  for img, ax in zip(image_array, axes):
    ax.imshow(np.array(img).reshape(IMG_SIZE, IMG_SIZE), cmap="gray")
  plt.tight_layout()
  plt.show()

def prepare(filepath):
  IMG_SIZE = 100
  img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
  new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
  return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

base_dir = "dataset/Cat_Dog_data"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

train_cats = os.path.join(train_dir, "cat")
train_dogs = os.path.join(train_dir, "dog")
test_cats = os.path.join(test_dir, "cat")
test_dogs = os.path.join(test_dir, "dog")


num_cats_tr = len(os.listdir(train_cats))
num_dogs_tr = len(os.listdir(train_dogs))
num_cats_ts = len(os.listdir(test_cats))
num_dogs_ts = len(os.listdir(test_dogs))

total_train = num_cats_tr + num_dogs_tr
total_test = num_cats_ts + num_dogs_ts

BATCH_SIZE = 32
IMG_SIZE = 100

train_image_generator = ImageDataGenerator(rescale=1./255)
test_image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(
    batch_size=BATCH_SIZE,
    directory=train_dir,
    color_mode="grayscale",
    shuffle=True,
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="binary"
)

test_data_gen = train_image_generator.flow_from_directory(
    batch_size=BATCH_SIZE,
    directory=test_dir,
    color_mode="grayscale",
    shuffle=True,
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="binary"
)

plot_images([train_data_gen[0][0][0] for i in range(1)])

model = Sequential([
                    Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
                    MaxPooling2D(2, 2),

                    Conv2D(64, (3, 3), activation="relu"),
                    MaxPooling2D(2, 2),

                    Conv2D(128, (3, 3), activation="relu"),
                    MaxPooling2D(2, 2),

                    Flatten(),
                    Dense(256, activation="relu"),

                    Dense(2, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

EPOCHS = 10

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
    epochs=EPOCHS,
    validation_data=test_data_gen,
    validation_steps=int(np.ceil(total_test / float(BATCH_SIZE)))
)