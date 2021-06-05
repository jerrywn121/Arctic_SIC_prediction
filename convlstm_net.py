import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        pad = kernel_size[0] // 2, kernel_size[1] // 2

        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=pad)

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
        return self.hidden_state


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, kernel_size):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = len(hidden_dim)

        layers = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            layers.append(ConvLSTMCell(input_dim=cur_input_dim,
                                       hidden_dim=self.hidden_dim[i],
                                       kernel_size=kernel_size))

        self.layers = nn.ModuleList(layers)
        self.conv_output = nn.Conv2d(self.hidden_dim[-1], output_dim, kernel_size=1)

    def forward(self, input_x, land_mask):
        assert len(input_x.shape) == 5
        input_frames = input_x.size(1)

        for t in range(input_frames):
            input_ = input_x[:, t]
            first_step = (t == 0)

            for layer_idx in range(self.num_layers):
                input_ = self.layers[layer_idx](input_, first_step=first_step)
        output = self.conv_output(input_)[:, None]
        output = torch.clamp(output, 0, 1)

        return output
