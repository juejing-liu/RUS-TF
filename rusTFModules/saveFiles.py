import tensorflow as tf 
import tensorflow.keras as keras
import pathlib
import os
# import matplotlib as plt 
import pandas as pd 
import numpy as np 
import json



def createBestFolder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        


def saveHistory(name, path, history):
    df = pd.DataFrame()

    for f, v in history:
        df[f] = v
    fileName = '{0}.csv'.format(name)
    path += fileName
    df.to_csv(path)
    
def modelToDisk(name, path, model):
    path += name
    os.mkdir(path)
    model.save(path)
    
def saveJson(name, path, modelPara):
    fileName = '{0}.json'.format(name)
    path += fileName
    with open(path, 'w') as f:
        json.dump(modelPara, f)