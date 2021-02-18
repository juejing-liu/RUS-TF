import json


# aList = [{'Dic1':1}, {'dic2':2}]

# with open('./rusTFModules/modelPara.json', 'w') as f:
#     json.dump(aList, f)

[{'layers':[{'layerName': 'dense', 'node': 32, 'activation': 'relu', 'input_shape':[1]},
            {'layerName': 'dense', 'node': 1, 'activation': None, 'input_shape':None}
           ],
   'optimize':{'optName':'RMSprop', 'parameters':0.001},
   'callBacks':[{'callName':'EarlyStopping', 'monitor': 'loss', 'patience': 10}],
   'compilePara': {'loss':"mean_squared_error", 'metrics':['mae', 'mse']},
   'epochs':500
}
]