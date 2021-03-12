import pandas as pd
from pandas.core.frame import DataFrame

import tensorflow as tf 
import tensorflow.keras as keras
import pathlib
import os
import matplotlib as plt 
import pandas as pd 
import numpy as np
from rusTFModules import dataProcess
import json
from rusTFModules import modelFunc
from rusTFModules import saveFiles


def generateDataSet(pathList, batch_size, number_epoch, train=True):
    dataNumber = 0
    labelArr = []
    featureArray = []
    for CSV in pathList:
        data = pd.read_csv(CSV, header=None)
        dataList = data[0].tolist()
        label = []
        label.append(dataList.pop(0))
        label.append(dataList.pop(0))
        labelArr.append(label)
        ref = round(dataList[0],6)
        itemNum = len(dataList)
        delElements = []
        for num in range (1, itemNum):
            element = dataList[num]
            comp = round(element,6)
            if (comp - ref ) == 0:
                delElements.append(element)
            else:
                ref = round(dataList[num], 6)
        processedData = [x for x in dataList if x not in delElements]
        
        featureArray.append(processedData[:50])
    
    arr = np.array(featureArray,dtype='float64')
    # features = tf.constant(featureArray)
    # labels = tf.constant(labelArr)
    labArr = np.array(labelArr, dtype='float32')

    dataSet =tf.data.Dataset.from_tensor_slices((arr, labelArr))
    dataSet = dataSet.shuffle(32)
    # dataSet = dataSet.map(pack_feature_vector)
    # print(dataSet)
    dataSet = dataSet.batch(batch_size, drop_remainder=False)
    # print(dataSet)
    # dataSet =dataSet.repeat(4)
    # DataNum = 0
    # for e in dataSet.as_numpy_iterator():
    #     print(e)
    #     DataNum += len(e[0])
    

    return {'dataSet': dataSet, 'dataNumber':len(pathList)}
        
def pack_feature_vector(features, labels):
    features = tf.stack(list(features.values()), axis=1)
    return features, labels


def rusDataProcess(trainDataPath, valDataPath, paraDict):
    np.set_printoptions(precision=15)
    trainCSV = []
    for root, dir, files in os.walk(trainDataPath, topdown=False):
        for name in files:
            trainCSV.append(os.path.join(root,name))
    dataDict = generateDataSet(trainCSV, batch_size=paraDict["batchSize"], number_epoch=paraDict["epochTime"])
    valCSV = []
    for root, dir, files in os.walk(valDataPath, topdown=False):
        for name in files:
            valCSV.append(os.path.join(root,name))
    valDataDict = generateDataSet(valCSV, batch_size=paraDict["batchSize"], number_epoch=paraDict["epochTime"], train=False)

    return {"trainDataset": dataDict['dataSet'], 
            "trainDataNum": dataDict['dataNumber'], 
            'valDataset': valDataDict['dataSet'],
            'valDataNum': valDataDict['dataNumber']}
    





 




