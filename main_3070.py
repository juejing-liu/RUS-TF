import tensorflow as tf 
import tensorflow.keras as keras
import pathlib
import os
# import matplotlib as plt 
import pandas as pd 
import numpy as np
# from rusTFModules import dataProcess
import json
# from rusTFModules import modelFunc
# from rusTFModules import saveFiles
# from rusTFModules import rusPreprocess
# import pickle
import trainFunc
from rusTFModules.configFiles.importDataset import datasetPara
from rusTFModules.configFiles.modelList import modelList


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
# tf.random.set_seed(42)

# with open('./rusTFModules/dataProcPara.json', 'r') as f:
#     paraDict = json.load(f)


# def removeKey(d, key):
#     r = dict(d)
#     del r[key]
#     return r







# test1.saveDataset('./saveDataset')
# test2 = loadDataset('./saveDataset')
# test2.loadFromFile()
# print(test1.trainDataSet)
# print(test1.trainDataNum)
# print(test1.valDataSet)
# print(test1.valDataNum)
# print(len(test1.trainDataSet.keys()))


# trainDataSet = tf.data.experimental.load('./rawData/trainDataSet/', tf.TensorSpec(shape=(32,50), dtype=tf.float64))
# print(trainDataSet)
# valDataSet = tf.data.experimental.load('./rawData/valDataSet/', tf.TensorSpec(shape=(32,50), dtype=tf.float64))

if __name__ == '__main__':
    scaleObj = trainFunc.modelScale()
    # print(scaleObj.mode)
    datasetPara.update(scaleObj=scaleObj, exam=False)
    test1 = trainFunc.dataObj(**datasetPara)
    test1.importData()
    for mod in modelList:
        m = trainFunc.mLearning(mod, dataObj=test1)
        m.checkSavePath()
        m.setLayers()
        m.setOptimizer()
        m.setCallBack()
        m.modelCompile()
        m.modelFit()
        m.fitHistory()
        m.saveModel()

# print(test1.trainDataSet)
# print(test1.valDataSet)

# def loadModel(path):
#     mod = keras.models.load_model(path)
#     return mod


# test1 = trainFunc.dataObj(**datasetPara)
# test1.importData()

# previousModel = loadModel('./result/test78')
# print(previousModel)
# print(test1.testDataset)
# testResult = previousModel.predict(test1.testDataset)
# testResult = np.array(testResult, dtype='float32')
# print(testResult)
# df = pd.DataFrame(testResult)
# df.to_csv('./testresult/predict_exp_64.csv')
# np.save('./testresult/predict_exp_64.npy', testResult)

# fileNum = 00
# for labels in test1.testDataset:
#     # data = np.array(labels)
#     print(labels[1])
#     df = pd.DataFrame(labels[1])
#     df.to_csv('./testresult/input{0}.csv'.format(fileNum))
#     fileNum =+1
    

    
# print(numpy_labels)













        
    
        