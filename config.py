import torch


class Configs:
    def __init__(self):
        pass


configs = Configs()

# trainer related
configs.n_cpu = 0
configs.device = torch.device('cuda:0')
configs.batch_size_test = 1
configs.tensorboard_path = "./tensorboard/"
configs.batch_size = 4
configs.lr = 0.001
configs.weight_decay = 0
configs.display_interval = 200
configs.num_epochs = 70
configs.early_stopping = True
configs.patience = 10
configs.gradient_clipping = True
configs.clipping_threshold = 1.

#data related

configs.input_dim = 1
configs.output_dim = 1
configs.input_length = 8
configs.output_length = 12
configs.input_gap = 1
configs.pred_shift = 12

# model related

configs.kernel_size = (3, 3)
configs.bias = True
configs.hidden_dim = (64, 64)
configs.d_attn = 32
configs.ssr_decay_rate = 0.8e-4