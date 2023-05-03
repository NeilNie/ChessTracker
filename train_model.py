import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
import matplotlib.pyplot as plt
from model.convnet import build_piece_classification_model


img_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0,
    rotation_range=0,
    # width_shift_range=0.15,
    # height_shift_range=0.15,
    vertical_flip=False,
    horizontal_flip=False,
    brightness_range=[0.75, 1.25],
    # validation_split=0.2,
)

img_gen_val = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0,
    rotation_range=0,
    # width_shift_range=0.15,
    # height_shift_range=0.15,
    vertical_flip=False,
    horizontal_flip=False,
    # validation_split=0.2,
)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    "./dataset_3/train",
    shuffle=True,
    batch_size=16,
    color_mode="grayscale",
    # labels=['empty', 'white', 'black'],
    label_mode='categorical',
    image_size=(100, 100))
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    "./dataset_3/val",
    shuffle=True,
    batch_size=1,
    color_mode="grayscale",
    # labels=['empty', 'white', 'black'],
    label_mode='categorical',
    image_size=(100, 100))


train_dataset = img_gen.flow_from_directory("./dataset_3/train", target_size=(100, 100), batch_size=16,
                                       color_mode ="grayscale", class_mode='categorical')
validation_dataset = img_gen_val.flow_from_directory("./dataset_3/val", target_size=(100, 100), batch_size=2,
                                        color_mode ="grayscale", class_mode='categorical')

print(len(train_dataset))


# for batch in train_dataset.as_numpy_iterator():
#     img, label = batch
#     print(label)

model = build_piece_classification_model(3)
model.summary()

model.compile(optimizer=optimizers.Adam(learning_rate=0.000030),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

history = model.fit(train_dataset, epochs=55,
                    validation_data=validation_dataset)

model.save("type_detector", save_format='h5')
print(history)
# plt.plot(history['accuracy'])
# plt.show()
