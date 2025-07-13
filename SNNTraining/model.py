import torch
import snntorch as snn
import snntorch.functional as SF
from snntorch import RSynaptic
import torch.nn as nn
from snntorch import surrogate


# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
        

 # /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class SpikingNet(torch.nn.Module):
    def __init__(self, opt, spike_grad=surrogate.fast_sigmoid(), learn_alpha=True, learn_beta=True, learn_treshold=True):
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

        spk1, syn1, mem1 = self.lif1.init_rsynaptic()
        mem2 = self.lif2.init_leaky()

        # Record the spikes from the hidden layer (if needed)
        spk1_rec = [] # not necessarily needed for inference
        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        if not time_first:
            #test = data
            x=x.transpose(1, 0)

        # Print the shape of the new tensor to verify the dimensions are swapped
        #print(x.shape)
        for step in range(self.num_steps):
            ## Input layer
            cur1 = self.fc1(x[step])

            ### Recurrent layer
            spk1, syn1, mem1 = self.lif1(cur1, spk1, syn1, mem1)

            ### Output layer
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)