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

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


class dataObj():

    def __init__(self, name, processFunc, trainDataPath, valDataPath, paraDict):
        self.name = name
        self.trainDataSet = None
        self.valDataSet = None
        self.processFunc = processFunc
        self.trainDataPath = trainDataPath
        self.valDataPath = valDataPath
        self.paraDict = paraDict

    def importData(self):
        # print ('begin')
        dataDict = self.processFunc(self.trainDataPath, self.valDataPath, self.paraDict)
        self.trainDataSet = dataDict["trainDatase"]
        self.valDataSet = dataDict["valDataset"]
        # print("done")

class mLearning():

    def __init__(self, dataObj, modelPara):
        self.dataObj = dataObj

        self.modelPara = modelPara
        self.callBack = None
        self.model = keras.Sequential
        self.optimizer = None


    def setLayers(self):
        for layer in self.modelPara['layers']:
            if layer['layerName'] == 'dense':
                layer = modelFunc.buildDense(node=layer['node'],
                                             activation=layer['activation'], 
                                             input_shape=layer['input_shape']
                )
            elif layer['layerName'] == 'Dropout':
                layer = modelFunc.buildDropout(layer['drop']
                )
        self.model.add(layer)
        
    def setOptimizer(self):
        self.optimizer = modelFunc.buildOptimizer(self.modelPara['optimize'])

    def setCallBack(self):
        # self.callBack = getattr(tf.keras.callbacks 
        pass
    def modelCompile(self):
        loss = self.modelPara['compilePara']['loss']
        metrics = self.modelPara['compilePara']['metrics']
        
        self.model.compile(loss=loss, metrics=metrics, optimizer=self.optimizer, )
    

# cwd = os.getcwd()
# print(cwd)


with open('./rusTFModules/dataProcPara.json', 'r') as f:
    paraDict = json.load(f)


test1 = dataObj(name='test', 
                processFunc=dataProcess.csvToData,
                trainDataPath='./examples/MLenthalpy/data.csv',
                valDataPath='./examples/MLenthalpy/val_data.csv',
                paraDict=paraDict)

test1.importData()

# print(test1.trainDataSet)
# print(test1.valDataSet)




        
    
        