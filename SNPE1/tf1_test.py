import tensorflow as tf
import numpy as np

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

model = tf.keras.models.load_model('models/tf1.h5')

loss,acc = model.evaluate(x=x_test,y=y_test, batch_size=128, verbose=1)
print('Acc = ', acc)

model.load_weights('checkpoints/tf1.10.ckpt')

loss,acc = model.evaluate(x=x_test,y=y_test, batch_size=128, verbose=1)
print('Epoch 10 Acc = ', acc)
