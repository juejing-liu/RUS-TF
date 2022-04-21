import numpy as np
import pickle


# arr1 = [[180,60], [220, 70], [300, 90]]
# arr1 = np.array(arr1, dtype=np.float32)

# # def mapCol(col):
# #     col = (col-180)*2
# #     return col+1


# # arr2 = np.apply_along_axis(mapCol, 0, arr1)

# # print(arr2)
# arr1[:,0] = (arr1[:,0] - 160) / (320 - 160) 
# print(arr1)

modelPath = './result/spectra_mode/test94'
scalePath = './result/spectra_mode/test94/_scale'
resultCSVPath = './testresult/predict_exp_64.csv'
with open(scalePath, 'rb') as f:
    scaleObj = pickle.load(f)
    
    
print(scaleObj.featureHigh)