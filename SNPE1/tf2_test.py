import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.cifar10.load_data()
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

model = tf.keras.models.load_model('models/tf2.h5')

model.summary()

tf.keras.utils.plot_model(model, to_file='tf2.png', show_shapes=True)

p_test = model.predict(x_test)
comp = np.argmax(p_test, axis=1) == np.argmax(y_test,axis=1)
print('Acc = ', sum(comp)/len(p_test))
print('confusion matrix = ')
print(confusion_matrix(np.argmax(y_test,axis=1), np.argmax(p_test, axis=1), labels=np.arange(10)))
