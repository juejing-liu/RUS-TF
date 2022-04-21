import tensorflow as tf


inputShape = (60, 64*64,1)
x = tf.random.normal(inputShape, dtype=tf.dtypes.float16)
# print(x)
conv1D_1 = tf.keras.layers.Conv1D(128, 16, strides=2, activation='relu',padding='same', input_shape = (10000, 1))(x)
# avg1D_1 = tf.keras.layers.AveragePooling1D()


print(conv1D_1)



