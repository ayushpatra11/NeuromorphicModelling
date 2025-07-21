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
        