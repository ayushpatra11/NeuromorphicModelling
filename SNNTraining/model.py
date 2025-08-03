####################################################################################
#
#   File Name: model.py
#   Author:  Ayush Patra
#   Description: This file contains the code for the SNN model and the metrics class.
#   Version History:        
#       - 2025-07-02: Initial version
#       - 2025-07-10: Added the code for training the SNN model based on previous
#                     version of code done by Aaron.
#       - 2025-07-21: Modified the code to seperate the forward for training phase
#                     and evaluation phase.
#       - 2025-07-26: Refactored the code to make it more readable and modular.
#
####################################################################################

import torch
import snntorch as snn
import snntorch.functional as SF
from snntorch import RSynaptic
import torch.nn as nn
from snntorch import surrogate


####################################################################################
#
#   Class Name: Metrics
#   Description: This class contains the metrics for the SNN model.
#   Version History:        
#       - 2025-07-02: Initial version based on the code done by Aaron.
#
####################################################################################

class Metrics():
    def __init__(self, ):
       
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0

    # return the predicted class from spike trains
    def return_predicted(self, output):
        
        _, predicted = output.sum(dim=0).max(dim=1)

        return predicted
   
    # calculate the confusion matrix
    def perf_measure(self, y_actual, y_hat):

        for i in range(len(y_hat)): 
            if y_actual[i]==y_hat[i]==1:
                self.TP += 1
            if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
                self.FP += 1
            if y_actual[i]==y_hat[i]==0:
                self.TN += 1
            if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
                self.FN += 1
        
    # calculate the precision
    def precision(self):
        return self.TP/(self.TP+self.FP+1e-8)

    # calculate the recall
    def recall(self):
        return self.TP/(self.TP+self.FN+1e-8)
    
    # calculate the harmonic mean of precision and recall
    def f1_score(self):
        return 2*(self.precision()*self.recall())/(self.precision()+self.recall()+1e-8)
    
    def get_scores(self):
        return self.TP, self.TN, self.FP, self.FN
        


####################################################################################
#
#   Class Name: Metrics
#   Description: This class contains the metrics for the SNN model.
#   Version History:        
#       - 2025-07-02: Initial version based on the code done by Aaron.
#       - 2025-07-10: Changed the code to seperate the forward for training phase
#                     and evaluation phase.
#       - 2025-07-21: Changed the code based on the bugs encountered during the
#                     training phase.
#       - 2025-07-21: Changed the code based on the bugs encountered during the
#                     evaluation phase and due to changes in NIR export.
#       - 2025-07-26: Refactored the code to make it more readable and modular.
#
####################################################################################

class SpikingNet(torch.nn.Module):
    def __init__(self, opt, spike_grad=surrogate.fast_sigmoid(), learn_alpha=True, learn_beta=True, learn_threshold=True):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(opt.num_inputs, opt.num_hidden1)
        self.fc1.__setattr__("bias",None) # biological plausability
        self.lif1 = RSynaptic(alpha=0.9, beta=0.9, spike_grad=spike_grad, learn_alpha=True, learn_threshold=True, linear_features=opt.num_hidden1, reset_mechanism="subtract", reset_delay=False, all_to_all=True)
        self.lif1.recurrent.__setattr__("bias",None) # biological plausability

        self.fc2 = nn.Linear(opt.num_hidden1, opt.num_outputs)
        self.fc2.__setattr__("bias",None) # biological plausability
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=spike_grad)

        self.num_steps = opt.num_steps

    def init_neurons():
        pass

    def forward_one_ts(self, x, spk1, syn1, mem1, mem2, cur_sub=None, cur_add=None, time_first=True):

        if not time_first:
            #test = data
            x=x.transpose(1, 0)
        curr_sub_rec = []
        curr_add_rec = []
        
        curr_sub_fc = []
        curr_add_fc = []
        if cur_sub is not None:
            for element in cur_sub:
                if element[2] > 99:
                    curr_sub_fc.append(element)
                    pass
                else:
                    curr_sub_rec.append(element)
                    pass

        if cur_add is not None:
            for element in cur_add:
                if element[2] > 99:
                    curr_add_fc.append(element)
                    pass
                else:
                    curr_add_rec.append(element)
                    pass

        ## Input layer
        cur1 = self.fc1(x)

        ### Recurrent layer
        spk1, syn1, mem1 = self.lif1(cur1, spk1, syn1, mem1)

        for element in curr_sub_rec:
            multiplier = element[0]
            w_idx = (element[2], element[1])
            cur_idx = element[2]

            weight = self.lif1.recurrent.weight.data[w_idx].item()

            syn1[cur_idx] = syn1[cur_idx] - weight*multiplier

        for element in curr_add_rec:
            multiplier = element[0]
            w_idx = (element[2], element[1])
            cur_idx = element[2]

            weight = self.lif1.recurrent.weight.data[w_idx].item()

            syn1[cur_idx] = syn1[cur_idx] + weight*multiplier

        ### Output layer
        cur2 = self.fc2(spk1)

        for element in curr_sub_fc:
            multiplier = element[0]
            w_idx = (element[2]-100, element[1])
            cur_idx = element[2]-100
            #print("WEIGHT DIMS", self.fc2.weight.data.shape)
            weight = self.fc2.weight.data[w_idx].item()

            cur2[cur_idx] = cur2[cur_idx] - weight*multiplier

        for element in curr_add_fc:
            multiplier = element[0]
            w_idx = (element[2]-100, element[1])
            cur_idx = element[2]-100

            weight = self.fc2.weight.data[w_idx].item()

            cur2[cur_idx] = cur2[cur_idx] + weight*multiplier

        spk2, mem2 = self.lif2(cur2, mem2)

        return spk2, spk1, syn1, mem1, mem2

    def forward(self, x, time_first=True):

        # DEBUG : THE CHANGES MADE HERE ARE BASED ON THE BUGS ENCOUNTERED DURING THE EVALUATION PHASE:
        # 1. The shape of the input x was not being checked.
        # 2. The shape of the spk1, syn1, mem1, mem2 was not being checked.
        # 3. The shape of the cur1, cur2 was not being checked.
        # 4. The shape of the spk2_rec, mem2_rec was not being checked.
        # 5. The shape of the spk1_rec was not being checked.

        # These changes are made on top of the classes defined by Aaron. These are because of the changes in 
        # SNNTorch library.

        
        #print("[DEBUG] Input x shape before transpose:", x.shape)

        if not time_first:
            # x: [batch, time, input] → [time, batch, input]
            x = x.permute(1, 0, 2).contiguous()

        batch_size = x.shape[1]

        spk1 = torch.zeros(batch_size, self.fc1.out_features, device=x.device)
        syn1 = torch.zeros_like(spk1)
        mem1 = torch.zeros_like(spk1)
        mem2 = torch.zeros(batch_size, self.fc2.out_features, device=x.device)

        #print("[DEBUG] Initial spk1 shape:", spk1.shape)
        # print("[DEBUG] Initial syn1 shape:", syn1.shape)
        # print("[DEBUG] Initial mem1 shape:", mem1.shape)
        # print("[DEBUG] Initial mem2 shape:", mem2.shape)

        # Record the spikes from the hidden layer (if needed)
        spk1_rec = [] # not necessarily needed for inference
        # Record the final layer
        spk2_rec = []
        mem2_rec = []


        assert x.shape[0] == self.num_steps, f"Expected time dimension {self.num_steps}, got {x.shape[0]}"
        assert x.shape[1] == batch_size, f"Expected batch size {batch_size}, got {x.shape[1]}"
        # print("[DEBUG] Input x shape after transpose:", x.shape)

        for step in range(self.num_steps):
            # print(f"[DEBUG] Step {step}: x[step] shape:", x[step].shape)
            ## Input layer
            cur1 = self.fc1(x[step])
            # print(f"[DEBUG] Step {step}: cur1 shape:", cur1.shape)

            ### Recurrent layer
            spk1, syn1, mem1 = self.lif1(cur1, spk1, syn1, mem1)
            # print(f"[DEBUG] Step {step}: spk1 shape after lif1:", spk1.shape)

            ### Output layer
            cur2 = self.fc2(spk1)
            # print(f"[DEBUG] Step {step}: cur2 shape:", cur2.shape)
            spk2, mem2 = self.lif2(cur2, mem2)
            # print(f"[DEBUG] Step {step}: spk2 shape after lif2:", spk2.shape)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)
            spk1_rec.append(spk1.clone()) 
            

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0), torch.stack(spk1_rec, dim=0)