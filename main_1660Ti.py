import tensorflow as tf 
import tensorflow.keras as keras
import pathlib
import os
# import matplotlib as plt 
import pandas as pd 
import numpy as np
from rusTFModules import dataProcess
import json
from rusTFModules import modelFunc
from rusTFModules import saveFiles
from rusTFModules import rusPreprocess

# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# for device in gpu_devices:
#     tf.config.experimental.set_memory_growth(device, True)
# tf.random.set_seed(42)


class dataObj():

    def __init__(self, name, processFunc, trainDataPath, valDataPath, testDataPath, paraDict, standardFile):
        self.name = name
        self.trainDataSet = None
        self.valDataSet = None
        self.standardFile = standardFile
        self.processFunc = processFunc
        self.trainDataPath = trainDataPath
        self.valDataPath = valDataPath
        self.paraDict = paraDict
        self.trainDataNum = 0
        self.valDataNum = 0
        self.testDataset = None
        self.testDataNum = 0
        self.testDataPath = testDataPath

    def importData(self):
        # print ('begin')
        dataDict = self.processFunc(self.trainDataPath, self.valDataPath, self.testDataPath, self.paraDict, self.standardFile)
        self.trainDataSet = dataDict["trainDataset"]
        self.valDataSet = dataDict["valDataset"]
        # self.trainDataSet = self.trainDataSet.repeat(self.paraDict['trianRepeat'])
        self.trainDataNum = dataDict['trainDataNum']
        self.valDataNum = dataDict['valDataNum']
        self.testDataset = dataDict['testDataset']
        self.testDataNum = dataDict['testDataNum']        
        # print(self.trainDataNum)
        # print(self.valDataNum)
        # print("done")

class mLearning():

    def __init__(self, modelPara, dataObj):
        self.dataObj = dataObj

        self.modelPara = modelPara
        self.callBack = []
        self.model = keras.Sequential()
        self.optimizer = None
        self.history = None


    def setLayers(self):
        for l in self.modelPara['layers']:
            layerName = l['layerName']
            layerPara = l['parameters']
            layer = modelFunc.buildLayer(layerName, layerPara)
            self.model.add(layer)
            
            # if l['layerName'] == 'dense':

            #     layer = modelFunc.buildDense(node=l['node'],
            #                                     activation=l['activation'], 
            #                                     input_shape=l['input_shape']
            #         )

            # elif l['layerName'] == 'Dropout':
            #     layer = modelFunc.buildDropout(l['drop']
            #     )
            # # print(layer)
            # self.model.add(layer)
        
    def setOptimizer(self):
        self.optimizer = modelFunc.buildOptimizer(self.modelPara['optimize'])

    def setCallBack(self):
        for callBack in self.modelPara['callBacks']:
            if callBack['callName'] == 'ModelCheckpoint':
                folderPath = self.modelPara['savePath'] + self.modelPara['name'] + '_best'
                saveFiles.createBestFolder(folderPath)
                para = callBack['parameters']
                
                path = os.path.join(folderPath, (self.modelPara['name'] + '_{epoch:04d}') )
                # pathName = (folderPath + "/"
                #             + self.modelPara['name']
                #           + '_{epoch:04d}'
                #         # + '_best'
                # )
                # path = self.modelPara['savePath'] + pathName
                para.update({'filepath': path})
                cb = modelFunc.buildCallback(callBack['callName'], para)
            else:
                cb = modelFunc.buildCallback(callBack['callName'], callBack['parameters'])
                
              
        
    def modelCompile(self):
        loss = self.modelPara['compilePara']['loss']
        metrics = self.modelPara['compilePara']['metrics']

        self.model.compile(loss=loss, 
                           metrics=metrics, 
                           optimizer=self.optimizer)
        self.model.summary()
        
    def modelFit(self):
        # print(self.dataObj.trainDataSet['x'])
        # print(self.dataObj.trainDataSet['y'])
        print(self.dataObj.valDataSet)
        self.history = self.model.fit(self.dataObj.trainDataSet,
                                      validation_data=self.dataObj.valDataSet,
                                      epochs=self.modelPara['epochs'],
                                    #   callbacks=self.callBack,
                                      steps_per_epoch=(self.dataObj.trainDataNum//self.dataObj.paraDict["batchSize"]),
                                      validation_steps=(self.dataObj.valDataNum//self.dataObj.paraDict["batchSize"]))
    
    def saveModel(self):
        name = self.modelPara['name']
        saveFiles.modelToDisk(name, self.modelPara['savePath'], self.model)
        saveFiles.saveJson(name, self.modelPara['savePath'], self.modelPara)
        # df = pd.DataFrame()

        # for f, v in self.history.history.items():
        #     df[f] = v
        # df.to_csv('./result/{0}.csv'.format(self.modelPara['name']))
    
    def saveModel(self):
        name = self.modelPara['name']
        saveFiles.modelToDisk(name,self.model)
        saveFiles.saveJson(self.modelPara['name'], self.modelPara)
        # os.mkdir('./result/{0}'.format(self.modelPara['name']))
        # self.model.save('./result/{0}'.format(self.modelPara['name']))
# cwd = os.getcwd()
# print(cwd)


with open('./rusTFModules/dataProcPara.json', 'r') as f:
    paraDict = json.load(f)


def removeKey(d, key):
    r = dict(d)
    del r[key]
    return r

test1 = dataObj(name='test', 
                processFunc=rusPreprocess.rusDataProcess,
                trainDataPath='./rawData/Train/',
                valDataPath='./rawData/Val/',
                testDataPath='./rawData/exam/',
                # standardFile='./rawData/Standard.txt',
                standardFile=None,
                paraDict=paraDict)

test1.importData()
# print(test1.trainDataSet)
# print(test1.trainDataNum)
# print(test1.valDataSet)
# print(test1.valDataNum)
# print(len(test1.trainDataSet.keys()))


# trainDataSet = tf.data.experimental.load('./rawData/trainDataSet/', tf.TensorSpec(shape=(32,50), dtype=tf.float64))
# print(trainDataSet)
# valDataSet = tf.data.experimental.load('./rawData/valDataSet/', tf.TensorSpec(shape=(32,50), dtype=tf.float64))

modelList = [
{'layers':[{'layerName': 'InputLayer', 'parameters':{'input_shape': [64]}},    
                        {'layerName': 'Dropout', 'parameters':{'rate': 0.1}},        
                        {'layerName': 'Dense', 'parameters':{'units': 32, 'activation': 'relu'}},
                        {'layerName': 'Dense', 'parameters':{'units': 16, 'activation': 'relu'}},
                        {'layerName': 'Dense', 'parameters':{'units': 8, 'activation': 'relu'}},
                        # {'layerName': 'dense', 'node': 1024, 'activation': 'relu', 'input_shape':None},        
                        {'layerName': 'Dense', 'parameters':{'units': 2, 'activation': None}}
                        ],
   'optimize':{'optName':'Adam', 'parameters':{'learning_rate': 0.001}},
   'callBacks':[{'callName':'ModelCheckpoint', 'parameters': {'moniter': 'val_loss', 'save_best_only': True, 'mode': 'auto'}},
                {'callName':'ReduceLROnPlateau', 'parameters': {'moniter': 'val_loss', 'factor': 0.2, 'patience': 10, 'min_lr': 0.0001}}],
                 # Path will generate automatically if use ModelCheckpoint
   'compilePara': {'loss': 'mean_squared_error', 'metrics':[ 'mae', 'mse']},
   'epochs':2048,
   'name': 'test72',
   'savePath': './result/',
   'comment': 'use adam  w/ small learning rate, 2048 epochs, 64 points, diff, repeat 8 times 32 16 8 dropout 0.1, find the best model decreasing learning rate'
}
]

for mod in modelList:
    m = mLearning(mod,dataObj=test1)
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


# previousModel = loadModel('./result/test23')
# print(previousModel)
# testResult = previousModel.predict(test1.testDataset, steps=4)
# testResult = np.array(testResult, dtype='float32')
# df = pd.DataFrame(testResult)
# df.to_csv('./testresult/predict_128I.csv')
# np.save('./testresult/predict.npy', testResult)
# fileNum = 0
# for labels in test1.testDataset:
#     # data = np.array(labels)
#     print(labels[1])
#     df = pd.DataFrame(labels[1])
#     df.to_csv('./testresult/input{0}.csv'.format(fileNum))
#     fileNum =+1
    

    
# print(numpy_labels)













        
    
        