from collections import deque

import torch
import torch.nn as nn

class PrototypeModel(nn.Module):
    def __init__(
        self,
        data_dim=1,
        seq_len=128,
        latent_dim=4,
        max_subseq_len=51,
        init_temperature=1.0,
        init_hard=False,
        reconstruction_loss_weight=1.0,
        time_loss_weight=0.0,
        time_gradient_scalar=0.1,
    ):
        super().__init__()

        assert max_subseq_len % 2 == 1, "max_subseq_len must be odd"
        self.conv_encoder = nn.Conv1d(data_dim, 20, kernel_size=21, padding='same')
        self.rnn_encoder = nn.GRU(input_size=20, hidden_size=20, bidirectional=True, batch_first=True)

        self.mlp_embed_encoder = StandardMLP(input_dim=data_dim, layer_sizes=(256,), output_dim=256)
        self.encoder_positional_encoding = nn.Parameter(torch.randn(1, seq_len, 256))
        self.decoder_positional_encoding = nn.Parameter(torch.randn(1, seq_len, latent_dim))
        self.transformer_encoder = nn.TransformerEncoderLayer(256, 4, 512, batch_first=True)
        self.delta_t_mlp = StandardMLP(input_dim=256, layer_sizes=(256,), output_dim=2)
        self.mlp_latent_encoder = StandardMLP(input_dim=256, layer_sizes=(256,), output_dim=latent_dim)
        self.mlp_decoder = StandardMLP(input_dim=latent_dim, layer_sizes=(256,), output_dim=data_dim)

        self.set_temperature(init_temperature)
        if init_hard:
            self.hard_sample()
        else:
            self.soft_sample()
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.max_subseq_len = max_subseq_len
        self.half_context_len = self.max_subseq_len // 2
        self.padding_len = self.half_context_len
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.time_loss_weight = time_loss_weight
        self.time_gradient_scalar = time_gradient_scalar

    def forward(self, traj):
        # traj is a tensor of shape (batch_size, seq_len, data_dim)
        assert traj.shape[1] == self.seq_len, "Trajectories must be of length {}".format(self.seq_len)
        batch_size = traj.shape[0]
        encoded_feats = self.mlp_embed_encoder(traj)
        encoded_feats = encoded_feats + self.encoder_positional_encoding
        delta_t_logits = self.delta_t_mlp(self.transformer_encoder(encoded_feats))[:, :-1, :]
        delta_t = nn.functional.gumbel_softmax(delta_t_logits, tau=self.temperature, hard=self.sample, dim=-1)[..., 1]
        delta_t.register_hook(lambda grad: grad * self.time_gradient_scalar)
        temporal_attention_weights = self.get_temporal_attention_weights(delta_t)

        encoded_feats = self.mlp_latent_encoder(encoded_feats)

        unfolded_feats = nn.Unfold(kernel_size=(self.max_subseq_len, 1), padding=(self.padding_len, 0))(encoded_feats.transpose(1, 2).unsqueeze(-1))
        unfolded_feats = unfolded_feats.reshape(batch_size, self.latent_dim, self.max_subseq_len, self.seq_len).permute(0, 3, 2, 1)
        attended_feats = torch.sum(unfolded_feats * temporal_attention_weights.unsqueeze(-1), dim=2)

        reconstructed_traj = self.mlp_decoder(attended_feats + self.decoder_positional_encoding)
        info = dict(
            delta_t_logits=delta_t_logits,
            delta_t=delta_t,
            temporal_attention_weights=temporal_attention_weights,
            latent_feats=attended_feats,
            reconstructed_traj=reconstructed_traj,
            ground_truth_traj=traj,
        )
        return reconstructed_traj, info

    def get_loss(self, traj):
        reconstructed_traj, info = self.forward(traj)
        reconstruction_loss = nn.functional.mse_loss(reconstructed_traj, traj)
        time_loss = info['delta_t'].mean()

        model_loss = self.reconstruction_loss_weight * reconstruction_loss + self.time_loss_weight * time_loss
        metrics = dict(
            loss=model_loss.item(),
            reconstruction_loss=reconstruction_loss.item(),
            time_loss=time_loss.item(),
            average_compression=1 / max(time_loss.item(), 1e-5),
        )

        return model_loss, metrics, info

    def set_temperature(self, temperature):
        self.temperature = temperature

    def hard_sample(self):
        self.sample = True

    def soft_sample(self):
        self.sample = False

    def get_temporal_attention_weights(self, delta_t):
        device = delta_t.device
        batch_size = delta_t.shape[0]
        seq_len = delta_t.shape[1] + 1
        padding = torch.ones(batch_size, self.padding_len, device=device)
        delta_t = torch.cat([padding, delta_t, padding], dim=1)
        attention_weights = deque([torch.ones(batch_size, seq_len, device=device)])

        forward_elapsed_t = torch.zeros(batch_size, seq_len, device=device)
        backward_elapsed_t = torch.zeros(batch_size, seq_len, device=device)

        base_indices = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, seq_len) + self.padding_len
        for i in range(self.half_context_len):
            forward_indices = base_indices + i
            backward_indices = base_indices - i - 1

            forward_elapsed_t = forward_elapsed_t + torch.gather(delta_t, 1, forward_indices)
            backward_elapsed_t = backward_elapsed_t + torch.gather(delta_t, 1, backward_indices)

            attention_weights.append(torch.maximum(1 - forward_elapsed_t, torch.zeros_like(forward_elapsed_t)))
            attention_weights.appendleft(torch.maximum(1 - backward_elapsed_t, torch.zeros_like(backward_elapsed_t)))

        attention_weights = torch.stack(list(attention_weights), dim=-1)
        attention_weights = attention_weights / torch.sum(attention_weights, dim=-1, keepdim=True)
        return attention_weights

def test_temporal_attention():
    l = 10
    model = PrototypeModel(seq_len=10, max_subseq_len=5)
    trajs = torch.randn(2, l, 1)
    model.get_loss(trajs)

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

if __name__ == '__main__':
    test_temporal_attention()
