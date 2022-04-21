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
import pickle
import trainFunc
from rusTFModules.configFiles.importDataset import datasetPara
from rusTFModules.configFiles.modelList import modelList

def loadModel(path):
    mod = keras.models.load_model(path)
    return mod



def reNormLabel(arr, scaleObj):
    c11Low = scaleObj.c11Low
    c11High = scaleObj.c11High
    c12Low = scaleObj.c12Low
    c12High = scaleObj.c12High
    try:
        c44High = scaleObj.c44High
        c44Low = scaleObj.c44Low
    except:
        c44High = None
        c44Low = None
        
    arr[:,0] = (arr[:,0] * (c11High-c11Low)) + c11Low
    arr[:,1] = (arr[:,1] * (c12High-c12Low)) + c12Low
    if c44High != None and c44Low != None:
        arr[:,2] = (arr[:,2] * (c44High-c44Low)) + c44Low
        
    return arr



if __name__ == '__main__':
    modelPath = './result/extra_models_steel_cylinder/test205_best/test205_0227'
    scalePath = './result/extra_models_steel_cylinder/test202/_scale'
    resultCSVPath = './testresult/predict_exp_64.csv'
    with open(scalePath, 'rb') as f:
        scaleObj = pickle.load(f)
    print(scaleObj.dataRange)
    # scaleObj.resolution = (16*16)
    # scaleObj.dataRange = [0.2,1.3]
    previousModel = loadModel(modelPath)
    datasetPara.update(scaleObj=scaleObj, exam=True)
    # print(datasetPara['modelScale'].c11Low)
    test1 = trainFunc.dataObj(**datasetPara)
    test1.importData()
    # print(len(test1.testDataset))
    testResult = previousModel.predict(test1.testDataset)
    testResult = np.array(testResult, dtype='float32')
    print(len(testResult))
    testResult = reNormLabel(testResult, scaleObj)
    print(testResult)

    df = pd.DataFrame(testResult)
    df.to_csv(resultCSVPath)
    










        
    
        