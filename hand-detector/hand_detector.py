import tensorflow as tf
import tf.keras as keras
from PIL import Image

class hand_detector:
    def __init__(model_path, input_size=(160, 160)):
        self.model = keras.models.load_model(model_path)
        self.input_size = input_size

    
    def predict(img_arr):
        # first resize
        img = Image.fromarray(img_arr.astype('uint8'), 'RGB')

        img = img.resize(self.input_size)

        dataset = tf.data.Dataset.from_tensor_slices([np.array(img)])

        predict = model.predict(testing_dataset.batch(1))[0, 0]

        return int(predict > 0)

