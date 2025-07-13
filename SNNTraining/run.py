from model import SpikingNet
from train_model import Trainer
from options import Variables
from options import Specs
from dataset import NavDataset as BinaryNavigationDataset
from sweep_handler import SweepHandler
import utils


from torch.utils.data import DataLoader
from snntorch import surrogate
import wandb
import torch


v = Variables()
s = Specs()

def sweep(config=None):
    config = config
    
    spike_grad = surrogate.sigmoid()
   
    net = SpikingNet(v, 
                        spike_grad=spike_grad)

    sample_data = torch.randn(v.num_steps, v.num_inputs)
    net = utils.init_network(net, sample_data)   

    # Create dataloader
    train_set = BinaryNavigationDataset(seq_len, n_in, recall_duration, p_group, input_f0, n_cues, t_cue, t_cue_spacing, n_input_symbols, length=100)
    val_set = BinaryNavigationDataset(seq_len, n_in, recall_duration, p_group, input_f0, n_cues, t_cue, t_cue_spacing, n_input_symbols, length=50)
    train_loader = DataLoader(train_set, batch_size=20, shuffle=True, num_workers=0)
    test_loader = DataLoader(val_set, batch_size=20, shuffle=True, num_workers=0)
    
    trainer = Trainer(net,
                    train_loader,
                    test_loader,
                    v.target_sparcity,
                    v.recall_duration,
                    num_epochs=v.num_epochs, 
                    learning_rate=0.0001, 
                    optimizer='Adam',
                    target_frequency=v.target_fr,
                    num_steps=v.num_steps,
                    wandb_logging=True)
    
    _, _ = trainer.train(v.device)


#def main():
print("\n#########################################")
print("#########################################\n")
print("!!IMPORTANT!! You need to create a WandB account and paste your authorization key in options.py to use this.")
print("\n#########################################")
print("#########################################\n")

sweep_handler = SweepHandler()
key=v.wandb_key
# wandb.login(key=key)
config = {'method': 'random'}
config['metric'] = sweep_handler.metric
config['parameters'] = sweep_handler.parameters_dict

#sweep_id = wandb.sweep(config, project="MScTheses")

# Parameters
n_in = v.num_inputs
t_cue_spacing = v.t_cue_spacing
recall_duration = v.recall_duration
seq_len = v.num_steps
v.num_steps = seq_len
batch_size = v.bs
input_f0 = 40. / 100.
p_group = v.p_group
n_cues = v.n_cues
t_cue = v.t_cue
n_input_symbols = 4

sweep()
# wandb.agent(sweep_id, sweep)
# wandb.finish()



# def sweep(config=None):
#     with wandb.init(config=config):
#         config = wandb.config
       
#         if config.surrogate_gradient == "atan":
#             spike_grad = surrogate.atan()

#         elif config.surrogate_gradient == "sigmoid":
#             spike_grad = surrogate.sigmoid()
        
#         elif config.surrogate_gradient == "fast_sigmoid":
#             spike_grad = surrogate.fast_sigmoid()

#         net = SpikingNet(v, 
#                          spike_grad=spike_grad, 
#                          learn_alpha=config.learn_alpha,
#                          learn_beta=config.learn_beta,
#                          learn_treshold=config.learn_threshold)

#         sample_data = torch.randn(v.num_steps, v.num_inputs)
#         net = utils.init_network(net, sample_data)   

#         # Create dataloader
#         train_set = BinaryNavigationDataset(seq_len, n_in, recall_duration, p_group, input_f0, n_cues, t_cue, t_cue_spacing, n_input_symbols, length=100)
#         val_set = BinaryNavigationDataset(seq_len, n_in, recall_duration, p_group, input_f0, n_cues, t_cue, t_cue_spacing, n_input_symbols, length=50)
#         train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=0)
#         test_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=True, num_workers=0)
        
#         trainer = Trainer(net,
#                         train_loader,
#                         test_loader,
#                         v.target_sparcity,
#                         v.recall_duration,
#                         num_epochs=v.num_epochs, 
#                         learning_rate=config.learning_rate, 
#                         optimizer=config.optimizer,
#                         target_frequency=v.target_fr,
#                         num_steps=v.num_steps,
#                         wandb_logging=True)
        
#         _, _ = trainer.train(v.device)



