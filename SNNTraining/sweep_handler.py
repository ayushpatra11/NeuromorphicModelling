####################################################################################
#
#   File Name: sweep_handler.py
#   Author:  Ayush Patra
#   Description: This file contains all the configuration and hyperparameters
#                to tune and train the model using sweep training and wandb
#   Version History:        
#       - 2025-07-02: Initial version
#       - 2025-07-10: Added the code for training the SNN model based on previous
#                     version of code done by Aaron.
#
####################################################################################

class SweepHandler():
    """
    This class is used to define the sweep parameters and the metric to be used for the sweep.
    """
    
    def __init__(self):

        self.metric = {
        'name': 'loss',
        'goal': 'minimize'   
        }

        self.parameters_dict = {
        'learning_rate': {
            'values': [0.001, 0.0001, 0.0001, 0.00001]
        },

        'optimizer': {
            'values': ['Adam', 'AdamW']
            },
        'learn_alpha': {
            'values': [True, False]
            },
        'learn_beta': { 
            'values': [True, False]
            },
        'learn_threshold': {
            'values': [True, False]
            },
        'surrogate_gradient': {
            'values': ["atan", "sigmoid", "fast_sigmoid"]
            },
        'batch_size':{
            'values':[10]
        }
        
            
        }
        