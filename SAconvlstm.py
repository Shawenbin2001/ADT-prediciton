import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torchinfo import summary
import torchvision.models as models


def attn(query, key, value):
    """
    attention 
    """
    scores = query.transpose(1, 2) @ key / math.sqrt(query.size(1))  # (N, S, S)
    attn = F.softmax(scores, dim=-1)
    output = attn @ value.transpose(1, 2)
    return output.transpose(1, 2)  # (N, C, S)


class SAAttnMem(nn.Module):
    def __init__(self, input_dim, d_model, kernel_size):
        """
        attention 
        """
        super().__init__()
        pad = kernel_size[0] // 2, kernel_size[1] // 2
        self.d_model = d_model
        self.input_dim = input_dim
        self.conv_h = nn.Conv2d(input_dim, d_model*3, kernel_size=1)
        self.conv_m = nn.Conv2d(input_dim, d_model*2, kernel_size=1)
        self.conv_z = nn.Conv2d(d_model*2, d_model, kernel_size=1)
        self.conv_output = nn.Conv2d(input_dim+d_model, input_dim*3, kernel_size=kernel_size, padding=pad)

    def forward(self, h, m):
        hq, hk, hv = torch.split(self.conv_h(h), self.d_model, dim=1)
        mk, mv = torch.split(self.conv_m(m), self.d_model, dim=1)
        N, C, H, W = hq.size()
        Zh = attn(hq.view(N, C, -1), hk.view(N, C, -1), hv.view(N, C, -1))  # (N, S, C)
        Zm = attn(hq.view(N, C, -1), mk.view(N, C, -1), mv.view(N, C, -1))  # (N, S, C)
        Z = self.conv_z(torch.cat([Zh.view(N, C, H, W), Zm.view(N, C, H, W)], dim=1))
        i, g, o = torch.split(self.conv_output(torch.cat([Z, h], dim=1)), self.input_dim, dim=1)
        i = torch.sigmoid(i)
        g = torch.tanh(g)
        m_next = i * g + (1 - i) * m
        h_next = torch.sigmoid(o) * m_next
        return h_next, m_next


class SAConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, d_attn, kernel_size):
        """
        saconvlstm
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        pad = kernel_size[0] // 2, kernel_size[1] // 2

        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=pad)
        self.sa = SAAttnMem(input_dim=hidden_dim, d_model=d_attn, kernel_size=kernel_size)

    def initialize(self, inputs):
        device = inputs.device
        batch_size, _, height, width = inputs.size()

        self.hidden_state = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        self.cell_state = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        self.memory_state = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)

    def forward(self, inputs, first_step=False):
        if first_step:
            self.initialize(inputs)

        combined = torch.cat([inputs, self.hidden_state], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        self.cell_state = f * self.cell_state + i * g
        self.hidden_state = o * torch.tanh(self.cell_state)
        # novel for sa-convlstm
        self.hidden_state, self.memory_state = self.sa(self.hidden_state, self.memory_state)
        return self.hidden_state


class SAConvLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, d_attn, kernel_size):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = len(hidden_dim)

        layers = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            layers.append(SAConvLSTMCell(input_dim=cur_input_dim,
                                         hidden_dim=self.hidden_dim[i],
                                         d_attn=d_attn,
                                         kernel_size=kernel_size))

        self.layers = nn.ModuleList(layers)

        self.convpool_intput = nn.Sequential(nn.Conv2d(1, 32, kernel_size=(3,3), stride=( 2, 2)),
                                             nn.ReLU(),
                                             nn.Conv2d(32, 64, kernel_size=(3,3), stride=(2,2)),
                                             nn.ReLU(),
                                             nn.Conv2d(64, 64, kernel_size=(3,3), stride=(2,2)),
                                             nn.ReLU(),
                                             nn.BatchNorm2d(64),
                                             nn.Conv2d(64, 1, kernel_size=(3,3), stride=(1,1), padding = (1,1))
                                             )
        self.convpool_test = nn.Sequential(nn.Conv2d(1, 32, kernel_size=(3,3), stride=( 2, 2)),
                                             nn.ReLU(),
                                             nn.Conv2d(32, 64, kernel_size=(3,3), stride=(2,2)),
                                             nn.ReLU(),
                                             nn.Conv2d(64, 1, kernel_size=(3,3), stride=(1,1), padding = 1))
        self.conv_output = nn.Conv2d(self.hidden_dim[-1], 1, kernel_size=1)        

        self.deconv = nn.Sequential(nn.ConvTranspose2d(1, 32, kernel_size=(4,4), stride=(2,2)),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(32, 64, kernel_size=(4,4), stride=(2,2)),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(64, 64, kernel_size=(3,3), stride=(2,2)),
                                    nn.ConvTranspose2d(64, 1, kernel_size=(1,1), stride=(1,1))
                                    )

    def forward(self, input_x, device=torch.device('cuda:0'), input_frames=8, future_frames=12, output_frames=19,
                teacher_forcing=False, scheduled_sampling_ratio=0, train=True):


        assert len(input_x.shape) == 5
        if train:
            if teacher_forcing and scheduled_sampling_ratio > 1e-6:

                teacher_forcing_mask = torch.bernoulli(scheduled_sampling_ratio *
                    torch.ones(input_x.size(0), future_frames - 1, 1, 1, 1))
            else:
                teacher_forcing = False
        else:
            assert input_x.size(1) == input_frames
            teacher_forcing = False

        total_steps = input_frames + future_frames - 1
        outputs = [None] * total_steps
        deconv_outputs = [None] * total_steps

        for t in range(total_steps):
            if t < input_frames:
                input_ = input_x[:, t].to(device)
                residual = input_.clone()
                conv_input_ = self.convpool_intput(input_)
            elif not teacher_forcing:
                input_ = deconv_outputs[t-1]
                residual = input_.clone()
                conv_input_ = self.convpool_intput(input_)
            else:
                mask = teacher_forcing_mask[:, t - input_frames].float().to(device)

                input_ = input_x[:, t].to(device) * mask + deconv_outputs[t-1] * (1 - mask)
                conv_input_ = self.convpool_intput(input_)
                residual = input_.clone()

            first_step = (t == 0)
            input_ = input_.float()

            first_step = (t == 0)
            for layer_idx in range(self.num_layers):
                lstm_input_ = self.layers[layer_idx](conv_input_, first_step=first_step)
                conv_input_ = lstm_input_

            if train or (t >= (input_frames - 1)): 
                outputs[t] = self.conv_output(lstm_input_)
                deconv_outputs[t] = self.deconv(outputs[t])
                
                deconv_outputs[t] = deconv_outputs[t][:,:,0:378,0:325] + residual

        outputs = [x for x in outputs if x is not None]
        deconv_outputs = [x for x in deconv_outputs if x is not None]

        if train:

            assert len(outputs) == output_frames
        else:

            assert len(outputs) == future_frames

        final_outputs = torch.stack(deconv_outputs, dim=1)
        return final_outputs 

