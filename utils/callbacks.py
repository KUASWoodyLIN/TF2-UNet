import os
import cv2
import numpy as np
import tensorflow as tf


class SaveModelOutput(tf.keras.callbacks.Callback):
    def __init__(self, data_list, log_dir):
        super(SaveModelOutput, self).__init__()
        self.data_list = data_list
        self.log_dir = log_dir
        gt_word = np.ones((30, 224, 3))
        pd_word = np.ones((30, 224, 3))
        cv2.putText(gt_word,
                    'Groud truth',
                    (22, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 0), 2)
        cv2.putText(pd_word,
                    'Predict',
                    (56, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 0), 2)
        self.add_word = tf.cast(np.hstack([gt_word, pd_word]), dtype=tf.float32)

    def on_train_begin(self, logs=None):
        path = os.path.join(self.log_dir, 'images')
        self.writer = tf.summary.create_file_writer(path)

    def on_epoch_end(self, epoch, logs=None):
        imgs_save = []
        for data in self.data_list:
            input_img = data['image']
            output_mask = self.model.predict(tf.expand_dims(input_img, axis=0))
            input_mask = data['mask']

            # change Gray to RGB
            input_mask = tf.one_hot(tf.cast(input_mask[:, :, 0], dtype=tf.int32), depth=3)
            output_mask = tf.argmax(output_mask[0], axis=-1)
            output_mask = tf.one_hot(output_mask, depth=3)
            # combine Image and mask
            input_combine = 0.65 * input_img + 0.35 * input_mask
            output_combine = 0.65 * input_img + 0.35 * output_mask
            image_combine = tf.concat([input_combine, output_combine], axis=1)
            image_combine = tf.concat([self.add_word, image_combine], axis=0)
            imgs_save.append(image_combine)
        # Save Image
        with self.writer.as_default():
            tf.summary.image("UNet output", imgs_save, max_outputs=16, step=epoch)
