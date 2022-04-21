from numpy import true_divide


datasetPara = {"name": "test",
               "trainDataPath": "F:/RUS_raw_data/High_entropy_alloy/train",
               "valDataPath": "F:/RUS_raw_data/High_entropy_alloy/val",
               "testDataPath": "F:/RUS_raw_data/High_entropy_alloy/exam",
               "paraDict": {"batchSize": 60, 
               "epochTime": 4, 
               "trianRepeat": 32,
               "dataRange": [0.1,1.3],
               "resolution": (256*256)},


}

labelNum = 3   # how many labels (C11 C12 C44)?

mode = 1   # 0: diff, 1: spectra, 2: raw

randomSeed = 42  # random seed using in the whole project,  
                 # e.g.choose test specta, drop points

#-----------------mode 0 & 2: diff. & rawdata method only-------------------

dataPoints = 64


#--------------------exam only--------------------

enableDelete = True
pointsToDelete = 4
cutoff = 64 


