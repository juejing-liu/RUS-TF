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
import random


def standardGen(filePath):
    CSV = filePath
    data = pd.read_csv(CSV, header=None)
    dataList = data[0].tolist()

    dataList.pop(0)
    dataList.pop(0)
    featureArray = []
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
    return processedData

def generateDataSet(pathList, batch_size, number_epoch, standardArr, different=True, train=True):
    dataNumber = 0
    labelArr = []
    featureArray = []
    for CSV in pathList:
        data = pd.read_csv(CSV, header=None)
        dataList = data[0].tolist()
        # print(dataList)
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
        res = next(x for x, val in enumerate(processedData) if val > 0.5)
        if standardArr != None and different == False:
            processedData = processedData[:512]
            diff = []
            zipObj = zip(processedData,standardArr[:512])
            for l1i, l2i in zipObj:
                diff.append(l1i-l2i)
        elif different == True:
            diff = []
            xArr = [*range(1,514)]
            diffArr = np.diff(processedData[:513])/np.diff(xArr)
            diff = [diffArr[i] for i in range((res + 1), (res + 129))]
            # diff = [x / max(diff) for x in diff]           
        
        featureArray.append(diff)
    
    arr = np.array(featureArray,dtype='float64')
    # features = tf.constant(featureArray)
    # labels = tf.constant(labelArr)
    labArr = np.array(labelArr, dtype='float32')

    dataSet = tf.data.Dataset.from_tensor_slices((arr, labelArr))
    if train == True:
        dataSet = dataSet.repeat(number_epoch)
        dataSet = dataSet.shuffle(128)
    # dataSet = dataSet.map(pack_feature_vector)
    # print(dataSet)
        dataSet = dataSet.batch(batch_size)
        
        dataNum = number_epoch*len(pathList)
    else:
        dataSet = dataSet.batch(batch_size)
        dataNum = len(pathList)
    print(dataSet)
    # 
    # DataNum = 0
    # for e in dataSet.as_numpy_iterator():
    #     print(e)
    #     DataNum += len(e[0])
    

    return {'dataSet': dataSet, 'dataNumber':dataNum}
        
        
        
def generateTestDataSet(pathList, batch_size, number_epoch, standardArr, different=True, train=True):
    dataNumber = 0
    labelArr = []
    featureArray = []
    random.seed(42)
    pathList = random.sample(pathList, 128)
    for CSV in pathList:
        data = pd.read_csv(CSV, header=None)
        dataList = data[0].tolist()
        # print(dataList)
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
        
        if standardArr != None and different == False:
            processedData = processedData[:512]
            diff = []
            zipObj = zip(processedData,standardArr[:512])
            for l1i, l2i in zipObj:
                diff.append(l1i-l2i)
        elif different == True:
            diff = []
            xArr = [*range(1,514)]
            diffArr = np.diff(processedData[:513])/np.diff(xArr)
            diff = [diffArr[i] for i in range(0, len(diffArr), 4)]
            maxNum = max(diff)
            diff = [x / maxNum for x in diff]           
        
        featureArray.append(diff)
    
    arr = np.array(featureArray,dtype='float64')
    # features = tf.constant(featureArray)
    # labels = tf.constant(labelArr)
    labArr = np.array(labelArr, dtype='float32')
    df = pd.DataFrame(labArr)
    # df.to_csv('./testresult/input_128I.csv')
    dataSet = tf.data.Dataset.from_tensor_slices((arr, labelArr))
    if train == True:
        dataSet = dataSet.repeat(4)
        dataSet = dataSet.shuffle(128)
    # dataSet = dataSet.map(pack_feature_vector)
    # print(dataSet)
        dataSet = dataSet.batch(batch_size)
        
        dataNum = 4*len(pathList)
    else:
        dataSet = dataSet.batch(batch_size)
        dataNum = len(pathList)
    print(dataSet)
    # 
    # DataNum = 0
    # for e in dataSet.as_numpy_iterator():
    #     print(e)
    #     DataNum += len(e[0])
    
    random.seed()
    return {'dataSet': dataSet, 'dataNumber':dataNum}
        
def pack_feature_vector(features, labels):
    features = tf.stack(list(features.values()), axis=1)
    return features, labels


def rusDataProcess(trainDataPath, valDataPath, testDataPath, paraDict, standPath=None, saveDataset=False):
    np.set_printoptions(precision=15)
    if standPath != None:
        standardArr = standardGen(standPath)
    else:
        standardArr = None
    
    
    trainCSV = []
    for root, dir, files in os.walk(trainDataPath, topdown=False):
        for name in files:
            trainCSV.append(os.path.join(root,name))
    dataDict = generateDataSet(trainCSV, batch_size=paraDict["batchSize"], number_epoch=paraDict["epochTime"], standardArr=standardArr)
    valCSV = []
    for root, dir, files in os.walk(valDataPath, topdown=False):
        for name in files:
            valCSV.append(os.path.join(root,name))
    valDataDict = generateDataSet(valCSV, batch_size=paraDict["batchSize"], number_epoch=paraDict["epochTime"], standardArr=standardArr, train=False)
    testCSV = []
    for root, dir, files in os.walk(testDataPath, topdown=False):
        for name in files:
            testCSV.append(os.path.join(root,name))
    testDataDict = generateTestDataSet(testCSV,batch_size=paraDict["batchSize"], number_epoch=paraDict["epochTime"], standardArr=standardArr, train=False)
    if saveDataset == True:
        tf.data.experimental.save(dataDict['dataSet'], trainDataPath)
        tf.data.experimental.save(valDataDict['dataSet'], valDataPath)
        with open('{0}dataNum.json'.format(trainDataPath), 'w') as f:
            json.dump({'dataNumber': dataDict['dataNumber']}, f)
        with open('{0}valDataNum.json'.format(valDataPath), 'w') as f:
            json.dump({'dataNumber': valDataDict['dataNumber']}, f)
    return {"trainDataset": dataDict['dataSet'], 
            "trainDataNum": dataDict['dataNumber'], 
            'valDataset': valDataDict['dataSet'],
            'valDataNum': valDataDict['dataNumber'],
            'testDataset': testDataDict['dataSet'],
            'testDataNum': testDataDict['dataNumber']}
    





 




