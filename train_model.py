import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
import matplotlib.pyplot as plt
from model.convnet import build_model


data_dir = "./dataset"
# dataset = tf.keras.preprocessing.image_dataset_from_directory(
#     data_dir, batch_size=64
# )

img_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    vertical_flip=False,
    horizontal_flip=True
)

img_data = img_gen.flow_from_directory(data_dir, target_size=(100, 100), batch_size=8)
print(len(img_data))

model = build_model(4)
model.summary()


model.compile(optimizer=optimizers.Adam(learning_rate = 0.00001),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

history = model.fit(img_data, epochs=5)

print(history)
# plt.plot(history['accuracy'])
# plt.show()
