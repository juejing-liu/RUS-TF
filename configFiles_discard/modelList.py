modelList = [

{'layers':[{'layerName': 'InputLayer', 'parameters':{'input_shape': [256]}},    
                        {'layerName': 'Dropout', 'parameters':{'rate': 0.1}}, 
                        # {'layerName': 'Dense', 'parameters':{'units': 64, 'activation': 'relu'}},


                        {'layerName': 'Dense', 'parameters':{'units': 96, 'activation': 'relu'}},
                        # {'layerName': 'dense', 'node': 1024, 'activation': 'relu', 'input_shape':None},        
                        {'layerName': 'Dense', 'parameters':{'units': 2, 'activation': None}}
                        ],
   'optimize':{'optName':'Adam', 'parameters':{'learning_rate': 0.001}},
   'callBacks':[{'callName':'ModelCheckpoint', 'parameters': {'moniter': 'val_loss', 'save_best_only': True, 'mode': 'auto'}},
                {'callName':'ReduceLROnPlateau', 'parameters': {'moniter': 'val_loss', 'factor': 0.2, 'patience': 10, 'min_lr': 0.000001}}
                ],
                 # Path will generate automatically if use ModelCheckpoint
   'compilePara': {'loss': 'mean_squared_error', 'metrics':[ 'mae', 'mse']},
   'epochs': 2048,
   'name': 'test98',
   'savePath': './result/spectra_mode/',
   'comment': 'use adam  w/ small learning rate, 2048 epochs, 16*16 points, spectra mode, from 0.2 to 1.3 MHz, repeat 4 times, dropout_0.1 96, find the best model, vary learning rate, 100k data, no delete points, normalized feature & label, pickle save scale'
}
]