import tensorflow as tf


def normalize(image, mask):
    image = tf.cast(image, tf.float32)/255.0
    mask -= 1
    return image, mask


@tf.function
def load_image_train(data, shape=(224, 224)):
    image = tf.image.resize(data['image'], shape)
    mask = tf.image.resize(data['segmentation_mask'], shape)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    mask = tf.round(mask)
    image, mask = normalize(image, mask)
    return image, mask


@tf.function
def load_image_test(data, shape=(224, 224)):
    image = tf.image.resize(data['image'], shape)
    mask = tf.image.resize(data['segmentation_mask'], shape)

    image, mask = normalize(image, mask)
    return image, mask