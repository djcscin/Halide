import tensorflow as tf
import numpy as np
import os

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=(784,)))
model.add(tf.keras.layers.Dense(1000, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(tf.keras.optimizers.SGD(),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])

if not os.path.isdir('checkpoints_mlp'):
    os.mkdir('checkpoints_mlp')

model.fit(x=x_train, y=y_train, batch_size=128, epochs=100,
          callbacks=[tf.keras.callbacks.TensorBoard(log_dir='./log_tf_mlp', write_grads=True),
                     tf.keras.callbacks.ModelCheckpoint('checkpoints_mlp/tf_mlp.{epoch:02}.ckpt', save_weights_only=True)],
                     verbose=1)

if not os.path.isdir('models'):
    os.mkdir('models')
model.save('models/tf_mlp.h5')