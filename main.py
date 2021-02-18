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
tf.random.set_seed(42)


class dataObj():

    def __init__(self, name, processFunc, trainDataPath, valDataPath, paraDict):
        self.name = name
        self.trainDataSet = None
        self.valDataSet = None
        self.processFunc = processFunc
        self.trainDataPath = trainDataPath
        self.valDataPath = valDataPath
        self.paraDict = paraDict
        self.trainDataNum = 0
        self.valDataNum = 0

    def importData(self):
        # print ('begin')
        dataDict = self.processFunc(self.trainDataPath, self.valDataPath, self.paraDict)
        self.trainDataSet = dataDict["trainDatase"]
        self.valDataSet = dataDict["valDataset"]
        self.trainDataSet = self.trainDataSet.repeat(self.paraDict['trianRepeat'])
        self.trainDataNum = dataDict['trainDataNum']
        self.valDataNum = dataDict['valDataNum']
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
            
            if l['layerName'] == 'dense':
                print(l)
                layer = modelFunc.buildDense(node=l['node'],
                                             activation=l['activation'], 
                                             input_shape=l['input_shape']
                )

            elif l['layerName'] == 'Dropout':
                layer = modelFunc.buildDropout(l['drop']
                )
            print(layer)
            self.model.add(layer)
        
    def setOptimizer(self):
        self.optimizer = modelFunc.buildOptimizer(self.modelPara['optimize'])

    def setCallBack(self):
        for callBack in self.modelPara['callBacks']:
            cb = getattr(tf.keras.callbacks,callBack['callName'])
            cb = cb(monitor=callBack['monitor'],patience=callBack['patience'])
            self.callBack.append(cb)
        
            
        
    def modelCompile(self):
        loss = self.modelPara['compilePara']['loss']
        metrics = self.modelPara['compilePara']['metrics']

        self.model.compile(loss=loss, 
                           metrics=metrics, 
                           optimizer=self.optimizer)
        self.model.summary()
        
    def modelFit(self):
        self.history = self.model.fit(self.dataObj.trainDataSet, 
                                      validation_data=self.dataObj.valDataSet,
                                      epochs=self.modelPara['epochs'],
                                    #   callbacks=self.callBack,
                                      validation_steps=int(np.ceil(self.dataObj.valDataNum/self.dataObj.paraDict['batchSize'])))
    
    def fitHistory(self):
        df = pd.DataFrame()

        for f, v in self.history.history.items():
            df[f] = v
        df.to_csv('./result/{0}.csv'.format(self.modelPara['name']))
    
    def saveModel(self):
        os.mkdir('./result/{0}'.format(self.modelPara['name']))
        self.model.save('./result/{0}'.format(self.modelPara['name']))
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


modelList = [{'layers':[{'layerName': 'dense', 'node': 16, 'activation': 'relu', 'input_shape':[1]},
            {'layerName': 'dense', 'node': 1, 'activation': None, 'input_shape':None}
           ],
   'optimize':{'optName':'RMSprop', 'parameters':0.001},
   'callBacks':[{'callName':'EarlyStopping', 'monitor': 'loss', 'patience': 10}],
   'compilePara': {'loss':"mean_squared_error", 'metrics':['mae', 'mse']},
   'epochs':512,
   'name': 'test3'
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




        
    
        