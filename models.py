import torch
import torch.nn as nn

class PrototypeModel(nn.Module):
    def __init__(
        self,
        latent_dim=4,
        positional_encoding_dim=4,
    ):
        super().__init__()

        self.conv_encoder = nn.Conv1d(1, 20, kernel_size=21)
        self.rnn_encoder = nn.GRU(input_size=20, hidden_size=20, bidirectional=True)

        self.delta_t_func = StandardMLP(input_dim=40, layer_sizes=(256, 256), output_dim=2)
        self.mlp_encoder = StandardMLP(input_dim=40, layer_sizes=(256, 256), output_dim=latent_dim)
        self.latent_dim = latent_dim

        self.mlp_decoder = StandardMLP(input_dim=latent_dim, layer_sizes=(256, 256), output_dim=1)

def get_activation(activation):
    if activation == 'elu':
        return nn.ELU
    elif activation == 'relu':
        return nn.ReLU
    else:
        return NotImplementedError("Activation not implemented yet")

class StandardMLP(nn.Module):
    def __init__(self, input_dim, layer_sizes=[400, 400, 400, 400], output_dim=1, activate_last=False, activation='elu'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_sizes = layer_sizes
        self.activation = get_activation(activation)

        if len(layer_sizes) == 0:
            if not activate_last:
                self.network = nn.Linear(input_dim, output_dim)
            else:
                self.network = nn.Sequential(nn.Linear(input_dim, output_dim), self.activation())
            return

        layer_list = [nn.Linear(self.input_dim, self.layer_sizes[0]), self.activation()]
        for i in range(len(self.layer_sizes) - 1):
            layer_list.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]))
            layer_list.append(self.activation())
        layer_list.append(nn.Linear(self.layer_sizes[-1], output_dim))
        if activate_last:
            layer_list.append(self.activation())

        self.network = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.network.forward(x)