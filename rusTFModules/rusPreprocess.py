# from trainFunc import dataObj
import pandas as pd
from pandas.core.frame import DataFrame

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
import random
from .configFiles import importDataset


# def standardGen(filePath):
#     CSV = filePath
#     data = pd.read_csv(CSV, header=None)
#     dataList = data[0].tolist()

#     dataList.pop(0)
#     dataList.pop(0)
#     featureArray = []
#     ref = round(dataList[0],6)
#     itemNum = len(dataList)
#     delElements = []
#     for num in range (1, itemNum):
#         element = dataList[num]
#         comp = round(element,6)
#         if (comp - ref ) == 0:
#             delElements.append(element)
#         else:
#             ref = round(dataList[num], 6)    
#     processedData = [x for x in dataList if x not in delElements]
#     return processedData

def generateDataSet(pathList, batch_size, number_epoch, modelScale, different=True, train=True):
    dataNumber = 0
    labelArr = []
    featureArray = []
    # print(pathList)
    for CSV in pathList:
        data = pd.read_csv(CSV, header=None)
        dataList = data[0].tolist()
        # print(dataList)
        label = []
        for i in range(importDataset.labelNum):
            label.append(dataList.pop(0))

        labelArr.append(label)
        # print(dataList)
        # ref = round(dataList[0],6) 
        # itemNum = len(dataList)
        # delElements = []
        # for num in range (1, itemNum):
        #     element = dataList[num]
        #     comp = round(element,6)
        #     if (comp - ref ) == 0:
        #         delElements.append(element)
        #     else:
        #         ref = round(dataList[num], 6)
        # processedData = [x for x in dataList if x not in delElements]
        processedData = dataList
        # res = next(x for x, val in enumerate(processedData) if val > 0.5)
        # if standardArr != None and different == False:
        #     processedData = processedData[:255]
        #     diff = []
        #     zipObj = zip(processedData,standardArr[:255])
        #     for l1i, l2i in zipObj:
        #         diff.append(l1i-l2i)
        if different == True:
            diff = []

            xArr = [*range(1,len(processedData)+1)]
            # print(xArr)
            # print(processedData)
            diff = np.diff(processedData)/np.diff(xArr)
            diffArr = diff.tolist()
            diff = [diffArr[i] for i in range((0), (importDataset.dataPoints))]
            # diff = [x / max(diff) for x in diff]           
        
        featureArray.append(diff)
    
    arr = np.array(featureArray,dtype='float64')
    # features = tf.constant(featureArray)
    # labels = tf.constant(labelArr)
    labArr = np.array(labelArr, dtype='float32')
    #normalize both features and labels here


    if train == True:
        arrAndScaleDict = normalizationFeature(arr, modelScale)  
        labArrAndScaleDict = normalizationLabel(labArr, 
                                                arrAndScaleDict['scaleObj'])
        dataSet = tf.data.Dataset.from_tensor_slices((arrAndScaleDict['array'], 
                                                      labArrAndScaleDict['array']))
        scaleObj = arrAndScaleDict['scaleObj']
        dataSet = dataSet.repeat(number_epoch)
        dataSet = dataSet.shuffle(128)
    # dataSet = dataSet.map(pack_feature_vector)
    # print(dataSet)
        dataSet = dataSet.batch(batch_size)
        
        dataNum = number_epoch*len(pathList)
    else:
        arrAndScaleDict = normalizationFeature(arr, modelScale, train=False)  
        labArrAndScaleDict = normalizationLabel(labArr, modelScale, train=False)
        dataSet = tf.data.Dataset.from_tensor_slices((arrAndScaleDict['array'], 
                                                      labArrAndScaleDict['array']))
        dataSet = dataSet.batch(batch_size, drop_remainder=True)
        dataNum = len(pathList)
        scaleObj = modelScale
        scaleObj.dataPoints = importDataset.dataPoints
    # print(dataSet)
    # dataSet = dataSet.map(lambda x, y: {'x': x, 'y': y})
    # print(dataSet)
    # 
    # DataNum = 0
    # for e in dataSet.as_numpy_iterator():
    #     print(e)
    #     DataNum += len(e[0])
    

    return {'dataSet': dataSet, 'dataNumber':dataNum, 'scaleObj': scaleObj}
        
        
        
def generateTestDataSet(pathList, batch_size, number_epoch, modelScale=None, different=True, train=True):
    dataNumber = 0
    labelArr = []
    featureArray = []
    random.seed(importDataset.randomSeed)
    pathList = random.sample(pathList, 128)
    for CSV in pathList:
        data = pd.read_csv(CSV, header=None)
        dataList = data[0].tolist()
        # print(dataList)
        label = []
        for i in range(importDataset.labelNum):
            label.append(dataList.pop(0))

        labelArr.append(label)

        # ref = round(dataList[0],6)
        # itemNum = len(dataList)
        # delElements = []
        # for num in range (1, itemNum):
        #     element = dataList[num]
        #     comp = round(element,6)
        #     if (comp - ref ) == 0:
        #         delElements.append(element)
        #     else:
        #         ref = round(dataList[num], 6)
        # processedData = [x for x in dataList if x not in delElements]
        # processedData.sort()
        processedData = dataList
        if importDataset.enableDelete == True:
            processedData = deletePoints(list=processedData,
                                    points=importDataset.pointsToDelete, 
                                    cutoff = importDataset.cutoff)
        # res = next(x for x, val in enumerate(processedData) if val > 0.5)
        # print(processedData[:66])
        
        # if standardArr != None and different == False:
        #     processedData = processedData[:64]
        #     diff = []
        #     zipObj = zip(processedData,standardArr[:64])
        #     for l1i, l2i in zipObj:
        #         diff.append(l1i-l2i)
        if different == True:
           diff = []
           xArr = [*range(1,len(processedData)+1)]
           diff = np.diff(processedData)/np.diff(xArr)
            
           diffArr = diff.tolist()
           diff = [diffArr[i] for i in range((0), (importDataset.dataPoints))]
            # maxNum = max(diff)
            # diff = [x / maxNum for x in diff]           
        
        featureArray.append(diff)
    
    arr = np.array(featureArray,dtype='float64')
    # features = tf.constant(featureArray)
    # labels = tf.constant(labelArr)
    labArr = np.array(labelArr, dtype='float32')
    df = pd.DataFrame(labArr)
    arrAndScaleDict = normalizationFeature(arr, modelScale, train=False)  
    labArrAndScaleDict = normalizationLabel(labArr, modelScale, train=False)
    # df.to_csv('./testresult/input_128I.csv')
    # print('normalizing exam dataset')
    # arrDict = normalizationFeature(arr,modelScale, train=False)
    # labArrDict = normalizationLabel(labArr, modelScale, train=False)
    # print(labArrAndScaleDict['array'])
    dataSet = tf.data.Dataset.from_tensor_slices((arrAndScaleDict['array'], labArrAndScaleDict['array']))
    # if train == True:
    #     dataSet = dataSet.repeat(4)
    #     dataSet = dataSet.shuffle(128)
    # # dataSet = dataSet.map(pack_feature_vector)
    # # print(dataSet)
    #     dataSet = dataSet.batch(batch_size)
        
    #     dataNum = 4*len(pathList)
    # else:
    dataSet = dataSet.batch(batch_size)
    dataNum = len(pathList)
    # print(dataSet)
    # dataSet = dataSet.map(lambda x, y: {'x': x, 'y': y})
    # print(dataSet)
    # DataNum = 0
    # for e in dataSet.as_numpy_iterator():
    #     print(e)
    #     DataNum += len(e[0])
    
    random.seed()
    return {'dataSet': dataSet, 'dataNumber':dataNum}
        
def pack_feature_vector(features, labels):
    features = tf.stack(list(features.values()), axis=1)
    return features, labels

def normalizationFeature(arr, modelScale, train=True):
    if train == True:
        # print(np.amin(arr))
        # print(np.amax(arr))
        low = np.amin(arr)
        high = np.amax(arr)
        print(low)
        print(high)
        # print(modelScale.featureLow)
        modelScale.featureLow = low
        modelScale.featureHigh = high
        scaleObj = modelScale
        

    else:
        low = modelScale.featureLow
        high = modelScale.featureHigh
        print(low)
        print(high)
        scaleObj = modelScale
    arr = (arr - low)/(high - low)  
    return {'array': arr, 'scaleObj': scaleObj}

def normalizationLabel(arr, modelScale, train=True):
    if train == True:
        c11Low = np.amin(arr[:,0])
        c11High = np.amax(arr[:,0])
        c12Low = np.amin(arr[:,1])
        c12High = np.amax(arr[:,1])
        modelScale.c11Low = c11Low
        modelScale.c11High = c11High
        modelScale.c12Low = c12Low
        modelScale.c12High = c12High
        try:
            c44Low = np.amin(arr[:,2])
            c44High = np.amax(arr[:,2])
            modelScale.c44Low = c44Low
            modelScale.c44High = c44High
        except:
            c44Low = None
            c44High = None
    else:
        c11Low = modelScale.c11Low
        c11High = modelScale.c11High
        c12Low = modelScale.c12Low
        c12High = modelScale.c12High
        try:
           c44Low = modelScale.c44Low
           c44High = modelScale.c44High
        except:
            c44Low = None
            c44High = None
        # print(c12Low)
    arr[:,0] = (arr[:,0]-c11Low)/(c11High-c11Low)
    arr[:,1] = (arr[:,1]-c12Low)/(c12High-c12Low)
    if c44High != None and c44Low != None:
        arr[:,2] = (arr[:,2]-c44Low)/(c44High-c44Low)
        
    return {'array': arr, 'scaleObj': modelScale}
    
# def resumeLabel(arr, modelScale):
#     arr[:,0] = (arr[:,0]*(370-170)) + 170
#     arr[:,1] = (arr[:,1]*(130-30)) + 30
#     return arr

def dataToSpectra(pathList, batch_size, number_epoch, modelScale, dataRange, resolution=(256*256), different=False, train=True):
    rangeDiff = dataRange[1] - dataRange[0]
    step = rangeDiff / resolution
    dataNumber = 0
    labelArr = []
    featureArray = []
    # print(pathList)
    for CSV in pathList:

        data = pd.read_csv(CSV, header=None)
        dataList = data[0].tolist()
        # print(dataList)
        label = []
        for i in range(importDataset.labelNum):
            label.append(dataList.pop(0))

        labelArr.append(label)
        # print(label)
        processedData = dataList
        spectrum = spectra(processedData, dataRange, step)
        # print(len(spectrum))
        featureArray.append(spectrum)
        # print(spectrum)

        if len(spectrum) < (resolution):   
            # print(CSV)
            # print(len(spectrum))
            # print(spectrum)
            spectrum.extend([0] * ((resolution)-len(spectrum)))
        # print(len(spectrum))
        # if len(spectrum) > (resolution):   
        #     print(CSV)
        #     print(len(spectrum))
        #     print(spectrum)
    arr = np.array(featureArray)
    labArr = np.array(labelArr)
    # print(arr.shape)
    # print(spectrum)
    if train == True:

        arrAndScaleDict = normalizationFeature(arr, modelScale)  
        labArrAndScaleDict = normalizationLabel(labArr, 
                                                arrAndScaleDict['scaleObj'])
        # print(arrAndScaleDict['array'])
        dataSet = tf.data.Dataset.from_tensor_slices((arrAndScaleDict['array'], 
                                                      labArrAndScaleDict['array']))
        scaleObj = arrAndScaleDict['scaleObj']
        dataSet = dataSet.repeat(number_epoch)
        dataSet = dataSet.shuffle(128)
    # dataSet = dataSet.map(pack_feature_vector)
    # print(dataSet)
        dataSet = dataSet.batch(batch_size)
        
        dataNum = number_epoch*len(pathList)
        scaleObj.resolution = resolution
        scaleObj.dataRange = dataRange
    else:

        arrAndScaleDict = normalizationFeature(arr, modelScale, train=False)  
        labArrAndScaleDict = normalizationLabel(labArr, modelScale, train=False)
        # print(arrAndScaleDict['array'])
        dataSet = tf.data.Dataset.from_tensor_slices((arrAndScaleDict['array'], 
                                                      labArrAndScaleDict['array']))
        dataSet = dataSet.batch(batch_size, drop_remainder=True)
        dataNum = len(pathList)
        scaleObj = modelScale
    return {'dataSet': dataSet, 'dataNumber':dataNum, 'scaleObj': scaleObj}


def dataToSpectraExam(pathList, batch_size, number_epoch, modelScale, dataRange, resolution, different=False, train=False):
    # print(resolution)
    random.seed(importDataset.randomSeed)
    if len(pathList) > 128:
        pathList = random.sample(pathList, 128)
    rangeDiff = dataRange[1] - dataRange[0]
    step = rangeDiff / resolution
    dataNumber = 0
    labelArr = []
    featureArray = []
    percentageDelete = []
    for CSV in pathList:

        data = pd.read_csv(CSV, header=None)
        dataList = data[0].tolist()
        # print(len(dataList))
        label = []
        for i in range(importDataset.labelNum):
            label.append(dataList.pop(0))

        labelArr.append(label)
         
        processedData = dataList

        if importDataset.enableDelete == True:

           
            processedData = deletePoints(list=processedData,
                                    points=importDataset.pointsToDelete, 
                                    cutoff = importDataset.cutoff)
            allPoint = sum(i<1.3 for i in processedData)
            print(allPoint)
            percentageDelete.append((importDataset.pointsToDelete/(allPoint+importDataset.pointsToDelete)))
        spectrum = spectra(processedData, dataRange, step)
        
        featureArray.append(spectrum)

        if len(spectrum) < (resolution):   
            # print(CSV)
            # print(len(spectrum))
            # print(spectrum)
            spectrum.extend([0] * ((resolution)-len(spectrum)))


    arr = np.array(featureArray, dtype='float64')
    labArr = np.array(labelArr, dtype='float32')

    df = pd.DataFrame(labArr)
    df.to_csv('./testresult/input_2.csv')
    arrAndScaleDict = normalizationFeature(arr, modelScale, train=False)  
    labArrAndScaleDict = normalizationLabel(labArr, modelScale, train=False)

    dataSet = tf.data.Dataset.from_tensor_slices((arrAndScaleDict['array'], labArrAndScaleDict['array']))

    dataSet = dataSet.batch(batch_size)
    dataNum = len(pathList)
    print(sum(percentageDelete)/len(percentageDelete))
    random.seed()
    return {'dataSet': dataSet, 'dataNumber':dataNum}

def rawDataTrain(pathList, batch_size, number_epoch, modelScale, train=False):
    
    dataNumber = 0
    labelArr = []
    featureArray = []
    # print(pathList)
    for CSV in pathList:
        data = pd.read_csv(CSV, header=None)
        dataList = data[0].tolist()
        # print(dataList)
        label = []
        for i in range(importDataset.labelNum):
            label.append(dataList.pop(0))

        labelArr.append(label)
        processedData = dataList
        processedData = [processedData[i] for i in range((0), (importDataset.dataPoints))]
        featureArray.append(processedData)
    arr = np.array(featureArray,dtype='float64')
    labArr = np.array(labelArr, dtype='float32')
    #normalize both features and labels here


    if train == True:
        arrAndScaleDict = normalizationFeature(arr, modelScale)  
        labArrAndScaleDict = normalizationLabel(labArr, 
                                                arrAndScaleDict['scaleObj'])
        dataSet = tf.data.Dataset.from_tensor_slices((arrAndScaleDict['array'], 
                                                      labArrAndScaleDict['array']))
        scaleObj = arrAndScaleDict['scaleObj']
        dataSet = dataSet.repeat(number_epoch)
        dataSet = dataSet.shuffle(128)
        dataSet = dataSet.batch(batch_size)
        
        dataNum = number_epoch*len(pathList)
    else:
        arrAndScaleDict = normalizationFeature(arr, modelScale, train=False)  
        labArrAndScaleDict = normalizationLabel(labArr, modelScale, train=False)
        dataSet = tf.data.Dataset.from_tensor_slices((arrAndScaleDict['array'], 
                                                      labArrAndScaleDict['array']))
        dataSet = dataSet.batch(batch_size, drop_remainder=True)
        dataNum = len(pathList)
        scaleObj = modelScale
        scaleObj.dataPoints = importDataset.dataPoints
    return {'dataSet': dataSet, 'dataNumber':dataNum, 'scaleObj': scaleObj}





def rawDataExam(pathList, batch_size, number_epoch, modelScale, train=False):
    
    dataNumber = 0
    labelArr = []
    featureArray = []
    random.seed(importDataset.randomSeed)
    if len(pathList) > 128:
        pathList = random.sample(pathList, 128)
    # print(len(pathList))
    for CSV in pathList:
        data = pd.read_csv(CSV, header=None)
        dataList = data[0].tolist()
        # print(dataList)
        label = []
        for i in range(importDataset.labelNum):
            label.append(dataList.pop(0))

        labelArr.append(label)
        processedData = dataList
        if importDataset.enableDelete == True:
            processedData = deletePoints(list=processedData,
                                    points=importDataset.pointsToDelete, 
                                    cutoff = importDataset.cutoff)
        processedData = [processedData[i] for i in range((0), (importDataset.dataPoints))]

        featureArray.append(processedData)
    # print(len(featureArray))    
    # print(len(labelArr))
    arr = np.array(featureArray,dtype='float64')
    labArr = np.array(labelArr, dtype='float32')
    df = pd.DataFrame(labArr)
    df.to_csv('./testresult/input_2.csv')
    #normalize both features and labels here



    arrAndScaleDict = normalizationFeature(arr, modelScale, train=False)  
    labArrAndScaleDict = normalizationLabel(labArr, modelScale, train=False)
    dataSet = tf.data.Dataset.from_tensor_slices((arrAndScaleDict['array'], 
                                                      labArrAndScaleDict['array']))
    dataSet = dataSet.batch(batch_size, drop_remainder=True)
    dataNum = len(pathList)
    print(len(arrAndScaleDict['array']))

    return {'dataSet': dataSet, 'dataNumber':dataNum}
  
  
def deletePoints(list, points, cutoff):
    random.seed(importDataset.randomSeed)
    for i in range(points):
        if len(list) > cutoff:
            list.pop(random.randrange(cutoff))
        else:
            list.pop(random.randrange(len(list)))  
    return list
  
def spectra(dataList, dataRange, step):   # range is a list or trupe, resolution is (a*b)
    currentPoint = dataRange[0] + step
    spectrum = []
    while currentPoint <= dataRange[1]:
        countPeaks = 0
        for i in dataList:
            if i <= currentPoint:
                countPeaks += 1
                dataList.remove(i)
            else:
                spectrum.append(countPeaks)
                break
        currentPoint += step
    return spectrum



def rusDataProcess(trainDataPath, valDataPath, testDataPath, paraDict, exam, modelScale, saveDataset=False,):
    np.set_printoptions(precision=15)
    # if standPath != None:
    #     standardArr = standardGen(standPath)
    # else:
    #     standardArr = None
    # print(modelScale.mode)
    if exam == False:
        trainCSV = []
        for root, dir, files in os.walk(trainDataPath, topdown=False):
            for name in files:
                trainCSV.append(os.path.join(root,name))
        # print('processing train dataset')
        # print(trainCSV)
        valCSV = []
        for root, dir, files in os.walk(valDataPath, topdown=False):
            for name in files:
                valCSV.append(os.path.join(root,name))
        if modelScale.mode == 0:

            dataDict = generateDataSet(trainCSV, 
                                   batch_size=paraDict["batchSize"], 
                                   number_epoch=paraDict["epochTime"],
                                   modelScale=modelScale
                                   )
        

        # print('processing val dataset')
            valDataDict = generateDataSet(valCSV, 
                                      batch_size=paraDict["batchSize"], 
                                      number_epoch=paraDict["epochTime"],
                                      modelScale=dataDict['scaleObj'], 
                                      train=False)
            testDataDict = {'dataSet': None, 'dataNumber': 0}
            scaleObj = valDataDict['scaleObj']
        
        if modelScale.mode == 1:
            print('spectra')
            dataDict = dataToSpectra(trainCSV, 
                                   batch_size=paraDict["batchSize"], 
                                   number_epoch=paraDict["epochTime"],
                                   modelScale=modelScale,
                                   dataRange=paraDict["dataRange"],
                                   resolution=paraDict["resolution"]
                                   )
            # print(valCSV)
            valDataDict = dataToSpectra(valCSV, 
                                      batch_size=paraDict["batchSize"], 
                                      number_epoch=paraDict["epochTime"],
                                      modelScale=dataDict['scaleObj'],
                                      dataRange=paraDict["dataRange"],
                                      resolution=paraDict["resolution"], 
                                      train=False)
            testDataDict = {'dataSet': None, 'dataNumber': 0}
            scaleObj = valDataDict['scaleObj']
        if modelScale.mode == 2:
            print('raw')
            dataDict = rawDataTrain(trainCSV, 
                                   batch_size=paraDict["batchSize"], 
                                   number_epoch=paraDict["epochTime"],
                                   modelScale=modelScale,
                                   train=True
                                   )
            # print(valCSV)
            valDataDict = rawDataTrain(valCSV, 
                                      batch_size=paraDict["batchSize"], 
                                      number_epoch=paraDict["epochTime"],
                                      modelScale=dataDict['scaleObj'],

                                      train=False)
            testDataDict = {'dataSet': None, 'dataNumber': 0}
            scaleObj = valDataDict['scaleObj']
    else:
        dataDict = {'dataSet': None, 'dataNumber': 0}
        valDataDict = {'dataSet': None, 'dataNumber': 0}
        testCSV = []
        for root, dir, files in os.walk(testDataPath, topdown=False):
            for name in files:
                testCSV.append(os.path.join(root,name))
        try:                        # early modelScale objs do't have mode attr 
            mode = modelScale.mode
        except:
            mode = 0
        if mode == 0:
            testDataDict = generateTestDataSet(testCSV,batch_size=paraDict["batchSize"], 
                                           number_epoch=paraDict["epochTime"],
                                           modelScale=modelScale,
                                           train=False)
        elif mode == 1:
            testDataDict = dataToSpectraExam(testCSV,batch_size=paraDict["batchSize"], 
                                           number_epoch=paraDict["epochTime"],
                                           modelScale=modelScale,
                                           dataRange=modelScale.dataRange,
                                           resolution=modelScale.resolution,
                                           train=False)
        elif mode == 2:
            print('raw_mode')
            testDataDict = rawDataExam(testCSV,batch_size=paraDict["batchSize"], 
                                           number_epoch=paraDict["epochTime"],
                                           modelScale=modelScale,
                                           train=False)
        
        scaleObj = modelScale
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
            'testDataNum': testDataDict['dataNumber'],
            'scaleObj': scaleObj}
    





 




