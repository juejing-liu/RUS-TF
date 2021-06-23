import tensorflow as tf 
import tensorflow.keras as keras
import pathlib
import os
import matplotlib as plt 
import pandas as pd 
import numpy as np 
import json


def saveHistory(name, history):
    df = pd.DataFrame()

    for f, v in history:
        df[f] = v
    df.to_csv('./result/{0}.csv'.format(name))
    
def modelToDisk(name, model):
    os.mkdir('./result/{0}'.format(name))
    model.save('./result/{0}'.format(name))
    
def saveJson(name, modelPara):
    with open('./result/{0}.json'.format(name), 'w') as f:
        json.dump(modelPara, f)