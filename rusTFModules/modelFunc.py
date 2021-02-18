import tensorflow as tf 
import tensorflow.keras as keras
import pathlib
import os
import matplotlib as plt 
import pandas as pd 
import numpy as np 


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

def buildDense(node, activation, input_shape):
    if (activation != None) and (input_shape != None):
        layer = keras.layers.Dense(node, activation=activation, input_shape=input_shape)
    elif(activation != None) and (input_shape == None):
        layer = keras.layers.Dense(node, activation=activation)
    else:
        layer = keras.layers.Dense(node)
         
    return layer

def buildDropout(drop):
    layer = keras.layers.Dropout(drop)
    return layer

def buildOptimizer(optiPara):
    name = optiPara['optName']
    para = optiPara['parameters']
    optimizer = getattr(tf.optimizers, name)
    optimizer = optimizer(para)
    return optimizer

# def buildModel():

