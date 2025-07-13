import torch

class Variables(object):
    def __init__(self):
        self.num_inputs = 20
        self.num_hidden1 = 100
        self.num_outputs = 2
        self.core_capacity = 25 # calculated automatically during mapping
        self.num_epochs = 100
        self.lr = 0.0001
        self.target_fr = 1.0
        self.bs = 10
        self.num_cores = 5
        self.target_sparcity = 1.0
        self.wandb_key = "paste-key-here"

        self.train = False

        self.recall_duration = 20
        self.t_cue_spacing = 15
        self.silence_duration = 30
        self.n_cues = 7
        self.t_cue = 10
        self.p_group = 0.3
        self.num_steps = int(self.t_cue_spacing *  self.n_cues + self.silence_duration + self.recall_duration)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Specs(object):
    def __init__(self):
        self.ADDR_W = 5
        self.MSG_W = 10
        self.EAST, self.NORTH, self.WEST, self.SOUTH, self.L1 = range(5)
        self.NUM_PACKETS_P_INJ = 20

        self.SID = 0b00001
        self.E_MASK = 0b10000
        self.N_MASK = 0b01000
        self.W_MASK = 0b00100
        self.S_MASK = 0b00010
