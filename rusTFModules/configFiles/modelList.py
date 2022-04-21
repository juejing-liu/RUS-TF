modelList = [

{'layers':[{'layerName': 'InputLayer', 'parameters':{'input_shape': [256*256]}},    
                              {'layerName': 'Dropout', 'parameters':{'rate': 0.1}},                          
                        # {'layerName': 'Dense', 'parameters':{'units': 64, 'activation': 'relu'}},
                        # {'layerName': 'Dense', 'parameters':{'units': 128, 'activation': 'relu'}},

                        {'layerName': 'Dense', 'parameters':{'units': 64, 'activation': 'softplus'}},
                        {'layerName': 'Dense', 'parameters':{'units': 16, 'activation': 'softplus'}},

                        # {'layerName': 'dense', 'node': 1024, 'activation': 'relu', 'input_shape':None},        
                        {'layerName': 'Dense', 'parameters':{'units': 2, 'activation': None}}
                        ],
   'optimize':{'optName':'Adam', 'parameters':{'learning_rate': 0.001}},
   'callBacks':[{'callName':'ModelCheckpoint', 'parameters': {'moniter': 'val_loss', 'save_best_only': True, 'mode': 'auto'}},
                {'callName':'ReduceLROnPlateau', 'parameters': {'moniter': 'val_loss', 'factor': 0.2, 'patience': 10, 'min_lr': 0.000000001}}
                ],
                 # Path will generate automatically if use ModelCheckpoint
   'compilePara': {'loss': 'mean_squared_error', 'metrics':[ 'mae', 'mse']},
   'epochs': 2048,
   'name': 'test301',
   'savePath': './result/extra_models_steel_cylinder/',
   'comment': 'Steel cylinder , 0.2 to 1.3 Mhz, 2 elastic modulars (labels), use adam  w/ small learning rate, 2048 epochs, spectra 16*16 input, repeat 4 times 48, dropout 0.1 64 16 2, activation function: softplus find the best model, vary learning rate 1e-2 to 1e-9, 100k data, no delete points, normalized feature & label, pickle save scale, '
},

]