import os
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from utils.models import create_unet_model
from utils.callbacks import SaveModelOutput
from utils.dataset import load_image_train, load_image_test


batch_size = 64
input_shape = (224, 224)

# Load dataset
dataset, info = tfds.load('oxford_iiit_pet', with_info=True,
                          data_dir='/home/share/dataset/tensorflow-datasets')

#  Train dataset
AUTOTUNE = tf.data.experimental.AUTOTUNE  # 自動調整模式
train = dataset['train'].map(lambda x: load_image_train(x, input_shape),
                             num_parallel_calls=AUTOTUNE)
train_dataset = train.cache().shuffle(1000).batch(batch_size)
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

#  Test dataset
test = dataset['test'].map(lambda x: load_image_test(x, input_shape))
test_dataset = test.batch(batch_size)

# testing images
test_images = [{'image':img, 'mask':mask} for img, mask in test.take(16)]

# Callbacks function
log_dirs = 'logs_unet_' + str(input_shape[0])
model_dirs = log_dirs + '/models'
os.makedirs(model_dirs, exist_ok=True)
model_tb = keras.callbacks.TensorBoard(log_dir=log_dirs)
model_save = keras.callbacks.ModelCheckpoint(model_dirs+'/unet.h5', monitor='val_loss')
model_smw = SaveModelOutput(test_images, log_dirs)

# Create model
model = create_unet_model(input_shape)
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[keras.metrics.CategoricalAccuracy()])

model.fit(train_dataset, epochs=100,
          validation_data=test_dataset,
          callbacks=[model_tb, model_save, model_smw])
