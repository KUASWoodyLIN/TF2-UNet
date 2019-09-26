import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow import keras
from utils.dataset import load_image_test


input_shape = (224, 224)

# Load dataset
dataset = tfds.load('oxford_iiit_pet', split=tfds.Split.TEST, data_dir='/home/share/dataset/tensorflow-datasets')

#  Test dataset
test = dataset.map(lambda x: load_image_test(x, input_shape))
TEST_IMAGES = [{'image':img, 'mask':mask} for img, mask in test.take(16)]

# Load model
model = keras.models.load_model('logs_unet_224/models/unet.h5')

h, w = 224, 448
images = np.zeros((h * 4, w * 4, 3))
for count, data in enumerate(TEST_IMAGES):
    input_img = data['image']
    output_mask = model.predict(tf.expand_dims(input_img, axis=0))
    input_mask = data['mask']

    # change Gray to RGB
    input_mask = tf.one_hot(tf.cast(input_mask[:, :, 0], dtype=tf.int32), depth=3)
    output_mask = tf.argmax(output_mask[0], axis=-1)
    output_mask = tf.one_hot(output_mask, depth=3)
    # combine Image and mask
    input_combine = 0.65 * input_img + 0.35 * input_mask
    output_combine = 0.65 * input_img + 0.35 * output_mask
    image_combine = tf.concat([input_combine, output_combine], axis=1)

    i = count // 4
    j = count % 4
    images[h * i:h * (i + 1), w * j:w * (j + 1)] = image_combine

plt.figure(dpi=400)
plt.title('Ground truth Mask / Predict Mask')
plt.axis('off')
plt.imshow(images)
plt.show()
plt.imsave('output_images/output.png', images)