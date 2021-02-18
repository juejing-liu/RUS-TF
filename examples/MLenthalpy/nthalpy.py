import tensorflow as tf 
import tensorflow.keras as keras
import pathlib
import os
import matplotlib as plt 
import pandas as pd 
import numpy as np 


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


# dataFile = tf.keras.utils.get_file('data.csv', './data.csv')
# valData = tf.keras.utils.get_custom_objects('val_data', './val_data.csv') 


# df = pd.read_csv('./data.csv', index_col=None)
# df.head()

# valDf = pd.read_csv('./val_data.csv', index_col=None)
# valDf.head()

# dfSlices = tf.data.Dataset.from_tensor_slices(dict(df))

# for featureBatch in dfSlices.take(1):
#     for key, value in featureBatch.items():
#         print("{0}: {1}".format(key, value))


dataBatches = tf.data.experimental.make_csv_dataset('./data.csv', batch_size=32, label_name='Cp', num_epochs=1)
valDataBatches = tf.data.experimental.make_csv_dataset('./val_data.csv', batch_size=32, label_name='Cp', num_epochs=1)
# for featureBatch, labelBatch in dataBatches.take(1):
#     print('Cp: {}'.format(labelBatch)),
#     print('Features:')
#     for key, value in featureBatch.items():
#         print("{0}: {1}".format(key, value))


# for featureBatch, labelBatch in valDataBatches.take(1):
#     print('Cp: {}'.format(labelBatch)),
#     print('Features:')
#     for key, value in featureBatch.items():
#         print("{0}: {1}".format(key, value))

def pack_feature_vector(features, labels):
    features = tf.stack(list(features.values()), axis=1)
    return features, labels


train_dataset = dataBatches.map(pack_feature_vector)
val_dataset = valDataBatches.map(pack_feature_vector)
train_dataset = train_dataset.repeat(28)

print(train_dataset)
print(val_dataset)



def build_model():
    model=keras.Sequential([keras.layers.Input(shape=[1]),
                            keras.layers.Dense(16, activation='relu'), 
    keras.layers.Dense(1)
    ])

    optimizer=tf.optimizers.RMSprop(0.001)

    model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=['mae', 'mse'])

    return model

model = build_model()
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

# class printDot(keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs):
#         if epoch%100 == 0: print('')
#         print('.', end='')

EPOCHS=500
history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, validation_steps=int(np.ceil(42/32)))

# acc = history.history['accuracy']

# val_acc = history.history['val_accuracy']

# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs_range = range(EPOCHS)
