import tensorflow as tf
import numpy as np
import os

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=(28,28,1)))
model.add(tf.keras.layers.Conv2D(32, (3,3), strides=(1,1),
                                 padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(tf.keras.layers.Reshape((14*14*32,)))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(tf.keras.optimizers.SGD(),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])

if not os.path.isdir('checkpoints'):
    os.mkdir('checkpoints')

model.fit(x=x_train, y=y_train, batch_size=128, epochs=100,
          callbacks=[tf.keras.callbacks.TensorBoard(log_dir='./log_tf1', write_grads=True),
                     tf.keras.callbacks.ModelCheckpoint('checkpoints/tf1.{epoch:02}.ckpt', save_weights_only=True)],
                     verbose=1)

if not os.path.isdir('models'):
    os.mkdir('models')
model.save('models/tf1.h5')
