
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import random   
import os



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


def listToArray(list, resolution):
    # print(resolution)
    array = []
    tensor = []
    for i in list:
        
        tensor.append(i)
        if len(tensor) == resolution[0]:
            # print(tensor)
            array.append(tensor)
            tensor = []
            
    return array

def normalization(spectrum, maxValue=6):
    newSpectrum = []
    for i in spectrum:
        i = i/maxValue
        newSpectrum.append(i)
    return newSpectrum 


dataRange = [0.05, 1.3]
resolution = [64,64]
step = (dataRange[1] - dataRange[0])/(resolution[1] * resolution[0])
dataPath = "F:/RUS_raw_data/High_entropy_alloy/train/fre_C11_100.000_C12_060.000_C44_010.000_mass_000.820_diameter_000.501_length_000.501_03.txt"


# trainCSV = []
# for root, dir, files in os.walk(dataPath, topdown=False):
#     for name in files:
#         trainCSV.append(os.path.join(root,name))

# random.seed(42)

# if len(trainCSV) > 128:
#    trainCSV = random.sample(trainCSV, 128)

# array = []


# for CSV in trainCSV:
    
data = pd.read_csv(dataPath, header=None)
   
dataList = data[0].tolist()
dataList.pop(0)
dataList.pop(0)
dataList.pop(0)
spectrum = spectra(dataList, dataRange, step)
print(spectrum)
spectrum = normalization(spectrum)
    
if len(spectrum) < (resolution[0]*resolution[1]):   
            # print(CSV)
            # print(len(spectrum))
            # print(spectrum)
    spectrum.extend([0] * ((resolution[0]*resolution[1])-len(spectrum)))

    
# print(array)


arr = np.array(spectrum)
print(spectrum)
# print(np.count_nonzero(arr))

font = {'family' : 'arial',
        'weight' : 'bold',

        'size'   : 24}

matplotlib.rc('font', **font)

x_ticks = [0, 32,64]
plt.xticks(x_ticks)

y_ticks = [0, 32,64]
plt.yticks(y_ticks)
ax = plt.gca()
ax.grid(which='minor', color='w', linestyle='-', linewidth=1)


# print(spectrum)
arr = listToArray(arr,resolution)
print(arr)
im = plt.imshow(arr, cmap='Greys', norm=matplotlib.colors.Normalize(0,0.35), extent=(0,64, 64,0), origin='upper')
plt.show()







            


