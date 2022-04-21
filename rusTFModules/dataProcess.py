import tensorflow as tf 
import tensorflow.keras as keras
import pathlib
import os
# import matplotlib as plt 
import pandas as pd 
import numpy as np 
import json

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)



def pack_feature_vector(features, labels):
    features = tf.stack(list(features.values()), axis=1)
    return features, labels



def csvToData(trainDataPath, valDataPath, paraDict):

    dataBatches = tf.data.experimental.make_csv_dataset(trainDataPath, 
                                                        batch_size=paraDict['batchSize'], 
                                                        label_name=paraDict['labelName'], 
                                                        num_epochs=paraDict['epochTime'])
    valDataBatches = tf.data.experimental.make_csv_dataset(valDataPath, 
                                                           batch_size=paraDict['batchSize'], 
                                                           label_name=paraDict['labelName'], 
                                                           num_epochs=paraDict['epochTime'])
    trainDataset = dataBatches.map(pack_feature_vector)
    valDataset = valDataBatches.map(pack_feature_vector)
    trainDataNum = 0
    valDataNum = 0
    for e in trainDataset.as_numpy_iterator():
        trainDataNum += len(e[0])
        
    for e in valDataset.as_numpy_iterator():
        valDataNum += len(e[0])

    return {"trainDatase": trainDataset, 
            "valDataset": valDataset, 
            'trainDataNum': trainDataNum,
            'valDataNum': valDataNum}