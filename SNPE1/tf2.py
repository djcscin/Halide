import tensorflow as tf
import numpy as np
import os

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.cifar10.load_data()
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

input1 = tf.keras.layers.Input(shape=(32,32,3))
conv1 = tf.keras.layers.Conv2D(32, (3,3), strides=(1,1),
                                 padding='same', activation='relu')(input1)
conv2 = tf.keras.layers.Conv2D(32, (3,3), strides=(1,1),
                                 padding='same', activation='relu')(conv1)
max1 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv2)
conv3 = tf.keras.layers.Conv2D(64, (3,3), strides=(1,1),
                                 padding='same', activation='relu')(max1)
conv4 = tf.keras.layers.Conv2D(64, (3,3), strides=(1,1),
                                 padding='same', activation='relu')(conv3)
max2 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv4)
conv5 = tf.keras.layers.Conv2D(128, (3,3), strides=(1,1),
                                 padding='same', activation='relu')(max2)
conv6 = tf.keras.layers.Conv2D(128, (3,3), strides=(1,1),
                                 padding='same', activation='relu')(conv5)
max3 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv6)
flat1 = tf.keras.layers.Flatten()(max3)
fc1 = tf.keras.layers.Dense(1000, activation='relu')(flat1)
fc2 = tf.keras.layers.Dense(10, activation='softmax')(fc1)

model = tf.keras.models.Model(inputs=[input1], outputs=[fc2])

model.compile(tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])

if not os.path.isdir('checkpoints_tf2'):
    os.mkdir('checkpoints_tf2')

model.fit(x=x_train, y=y_train, validation_data=[x_test, y_test], batch_size=128, epochs=100,
          callbacks=[tf.keras.callbacks.TensorBoard(log_dir='./log_tf2', write_grads=True),
                     tf.keras.callbacks.ModelCheckpoint('checkpoints_tf2/tf2.{epoch:02}.ckpt', save_weights_only=True)],
                     verbose=1)

if not os.path.isdir('models'):
    os.mkdir('models')
model.save('models/tf2.h5')