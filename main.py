import tensorflow_datasets
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ds_train = tensorflow_datasets.load(name='rock_paper_scissors',split="train")
ds_test = tensorflow_datasets.load(name='rock_paper_scissors',split="test")

train_images =np.array([example['image'].numpy()[:,:,0] for example in ds_train])
train_labels = np.array([example['label'].numpy() for example in ds_train])
test_images =np.array([example['image'].numpy()[:,:,0] for example in ds_test])
test_labels = np.array([example['label'].numpy() for example in ds_test])

train_images = train_images.reshape(2520, 300,300,1)
test_images = test_images.reshape(372, 300,300,1)

train_images =train_images.astype('float32')
test_images =test_images.astype('float32')

train_images /=255
test_images /=255


model = keras.Sequential([
   keras.layers.Conv2D(64, 3, activation='relu', input_shape=(300,300,1)),
   keras.layers.Conv2D(32, 3, activation='relu'),
   keras.layers.Flatten(),
   keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5, batch_size=2)