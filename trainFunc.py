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
from rusTFModules.rusPreprocess import rusDataProcess
import pickle
from rusTFModules.configFiles import importDataset


class modelScale():
    def __init__(self, featureLow=None, featureHigh=None, c11Low=None, c11High=None, 
                 c12Low=None, c12High=None, c44High=None, c44Low=None, dataRange=None, 
                 resolution=None, mode=importDataset.mode, # works strange
                 dataPoints=None):
        self.featureLow = featureLow
        self.featureHigh = featureHigh
        self.c11Low = c11Low
        self.c11Low = c11High
        self.c12Low = c12Low
        self.c12High = c12High
        self.c44Low = c44Low
        self.c44High = c44High
        self.dataRange = dataRange
        self.resolution = resolution
        self.dataPoints = dataPoints
        self.mode = mode # 0: diff, 1: spectra, 2: raw
                
class dataObj():

    def __init__(self, name, trainDataPath, valDataPath, testDataPath, paraDict, scaleObj, exam=False):
        self.name = name
        self.trainDataSet = None
        self.valDataSet = None
        # self.standardFile = standardFile
        # self.processFunc = rusPreprocess
        self.trainDataPath = trainDataPath
        self.valDataPath = valDataPath
        self.paraDict = paraDict
        self.trainDataNum = 0
        self.valDataNum = 0
        self.testDataset = None
        self.testDataNum = 0
        self.testDataPath = testDataPath
        # self.modelScale = modelScale
        self.exam = exam
        self.scaleObj = scaleObj
        # print(self.scaleObj.c11Low)

    def importData(self):
        # print ('begin')
        # print(self.trainDataPath)
        # self.scaleObj = modelScale()
        


        dataDict = rusDataProcess(self.trainDataPath, 
                                  self.valDataPath, 
                                  self.testDataPath, 
                                  self.paraDict, 
                                  self.exam, 
                                  self.scaleObj)

        self.trainDataSet = dataDict["trainDataset"]
        self.valDataSet = dataDict["valDataset"]
        # self.trainDataSet = self.trainDataSet.repeat(self.paraDict['trianRepeat'])
        self.trainDataNum = dataDict['trainDataNum']
        self.valDataNum = dataDict['valDataNum']
        self.testDataset = dataDict['testDataset']
        self.testDataNum = dataDict['testDataNum']        
        self.scaleObj = dataDict['scaleObj']
        print(self.scaleObj.c11Low)

    def saveDataset(self, path):
        trainCatch = os.path.join(path, 'trainCatch')
        # trainLabel = os.path.join(path, 'trainLabel')
        valCatch = os.path.join(path, 'valCatch')
        # valLabel = os.path.join(path, 'valLabel')        
        examCatch = os.path.join(path, 'examCatch')
        # examLabel = os.path.join(path, 'examLabel')
        datasetNumPath = os.path.join(path, 'datasetNumber.json')
        datasetParaDictPath = os.path.join(path, 'datasetParaDict.json')
        
        # names = list(self.trainDataSet.element_spec.keys())
        # print(names)
        splitTrain = splitDatasets(self.trainDataSet)
        splitVal = splitDatasets(self.valDataSet)
        splitExam = splitDatasets(self.testDataset)
        # tf.data.experimental.save(splitTrain['x'], trainFeature)
        # tf.data.experimental.save(splitTrain['y'], trainLabel)     
        # tf.data.experimental.save(splitVal['x'], valFeature)
        # tf.data.experimental.save(splitVal['y'], valLabel)
        # tf.data.experimental.save(splitExam['x'], examFeature)     
        # tf.data.experimental.save(splitExam['y'], examLabel)
        datasetPara = {'trainDataNum': self.trainDataNum,
                    #    'trainSpec': .element_spec,
                       'valDataNum': self.valDataNum,
                    #    'valSpec': self.valDataSet.element_spec,
                       'testDataNum': self.testDataNum,
                    #    'testSpec': self.testDataset.element_spec,
                       'name': self.name}
        with open(datasetNumPath, 'w') as f:
            json.dump(datasetPara, f)
        with open(datasetParaDictPath, 'w') as f:
            json.dump(self.paraDict ,f)


def splitDatasets(dataset):
    names = list(dataset.element_spec.keys())
    # print(names)
    tensors = {}
    for name in names:
    
        tensors[name] = dataset.map(lambda x: x[name])

    return tensors


        
class mLearning():

    def __init__(self, modelPara, dataObj, ifLoad=False):
        self.dataObj = dataObj

        self.modelPara = modelPara
        self.callBack = []
        self.model = keras.Sequential()
        self.optimizer = None
        self.history = None
        self.ifLoad = ifLoad

    def checkSavePath(self):
        saveFiles.createBestFolder(self.modelPara['savePath'])
        
        
        
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
                
            
            # cb = getattr(tf.keras.callbacks,callBack['callName'])
            # cb = cb(monitor=callBack['monitor'],patience=callBack['patience'])
            self.callBack.append(cb)
        
            
        
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
        # print(self.dataObj.valDataSet)
        if self.ifLoad == False:
            # self.dataObj.trainDataSet.catch()
            self.history = self.model.fit(self.dataObj.trainDataSet,
                                        validation_data=self.dataObj.valDataSet,
                                        epochs=self.modelPara['epochs'],
                                        callbacks=self.callBack,
                                        steps_per_epoch=(self.dataObj.trainDataNum//self.dataObj.paraDict["batchSize"]),
                                        validation_steps=(self.dataObj.valDataNum//self.dataObj.paraDict["batchSize"]))
        else:
            trainDataset = tf.data.Dataset.zip((self.dataObj.trainFeature, self.dataObj.trainLabel))
            valDataset = tf.data.Dataset.zip((self.dataObj.valFeature, self.dataObj.valLabel))
            self.history = self.model.fit(trainDataset,
                                        validation_data=valDataset,
                                        epochs=self.modelPara['epochs'],
                                        callbacks=self.callBack,
                                        steps_per_epoch=((self.dataObj.trainDataNum//self.dataObj.paraDict["batchSize"])+1),
                                        validation_steps=((self.dataObj.valDataNum//self.dataObj.paraDict["batchSize"])+1)
            )           
    
    def fitHistory(self):
        his = self.history.history.items()
        name = self.modelPara['name']
        saveFiles.saveHistory(name, self.modelPara['savePath'], his)
        # df = pd.DataFrame()

        # for f, v in self.history.history.items():
        #     df[f] = v
        # df.to_csv('./result/{0}.csv'.format(self.modelPara['name']))
    
    def saveModel(self):
        name = self.modelPara['name']
        saveFiles.modelToDisk(name, self.modelPara['savePath'], self.model)
        
        saveFiles.saveJson(name, self.modelPara['savePath'], self.modelPara)
        # self.dataObj.scaleObj.saveScale(self.modelPara['savePath'], name)
        with open(os.path.join(self.modelPara['savePath'], name, '_scale'), 'wb') as f:
            pickle.dump(self.dataObj.scaleObj, f)
        # os.mkdir('./result/{0}'.format(self.modelPara['name']))
        # self.model.save('./result/{0}'.format(self.modelPara['name']))
# cwd = os.getcwd()
# print(cwd)

