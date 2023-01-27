from collections import deque

import numpy as np
import torch
import torch.nn as nn

class PrototypeModel(nn.Module):
    def __init__(
        self,
        data_dim=1,
        seq_len=128,
        latent_dim=4,
        positional_encoding_dim=128,
        transformer_encoder_layers=2,
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

        self.mlp_embed_encoder = StandardMLP(input_dim=data_dim, layer_sizes=(256,), output_dim=128)
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_len, positional_encoding_dim))
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(128 + positional_encoding_dim, 4, 512, batch_first=True), transformer_encoder_layers)
        self.delta_t_mlp = StandardMLP(input_dim=128 + positional_encoding_dim, layer_sizes=(256,), output_dim=2)
        self.mlp_latent_encoder = StandardMLP(input_dim=128 + positional_encoding_dim, layer_sizes=(256,), output_dim=latent_dim)
        self.mlp_decoder = StandardMLP(input_dim=latent_dim + positional_encoding_dim + max_subseq_len, layer_sizes=(256,), output_dim=data_dim)

        self.set_temperature(init_temperature)
        if init_hard:
            self.hard_sample()
        else:
            self.soft_sample()
        self.latent_dim = latent_dim
        self.positional_encoding_dim = positional_encoding_dim
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
        encoded_feats = torch.cat([encoded_feats, self.positional_encoding.expand(batch_size, self.seq_len, self.positional_encoding_dim)], dim=-1)
        delta_t_logits = self.delta_t_mlp(self.transformer_encoder(encoded_feats))[:, :-1, :]
        delta_t = nn.functional.gumbel_softmax(delta_t_logits, tau=self.temperature, hard=self.sample, dim=-1)[..., 1]
        if delta_t.requires_grad:
            delta_t.register_hook(lambda grad: grad * self.time_gradient_scalar)
        temporal_attention_weights = self.get_temporal_attention_weights(delta_t)

        encoded_feats = self.mlp_latent_encoder(encoded_feats)

        unfolded_feats = nn.Unfold(kernel_size=(self.max_subseq_len, 1), padding=(self.padding_len, 0))(encoded_feats.transpose(1, 2).unsqueeze(-1))
        unfolded_feats = unfolded_feats.reshape(batch_size, self.latent_dim, self.max_subseq_len, self.seq_len).permute(0, 3, 2, 1)
        attended_feats = torch.sum(unfolded_feats * temporal_attention_weights.unsqueeze(-1), dim=2)

        reconstructed_traj = self.mlp_decoder(torch.cat([attended_feats, self.positional_encoding.expand(batch_size, self.seq_len, self.positional_encoding_dim), temporal_attention_weights], dim=-1))
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

    def set_time_loss_weight(self, time_loss_weight):
        self.time_loss_weight = time_loss_weight

    def hard_sample(self):
        self.sample = True

    def soft_sample(self):
        self.sample = False

    def train(self, mode=True):
        if mode:
            self.soft_sample()
        else:
            self.hard_sample()
        super().train(mode)

    def eval(self):
        self.hard_sample()
        super().eval()

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

class FullPrototypeModel(nn.Module):
    def __init__(self, cfg, data_dim, max_seq_len):
        super().__init__()
        self.cfg = cfg
        self.data_dim = data_dim
        self.max_seq_len = max_seq_len
        self.sample = False

        self.temperature = cfg.init_temperature
        self.time_loss_weight = cfg.time_loss_weight

        self.encoder = StandardMLP(input_dim=data_dim, **cfg.encoder_params, output_dim=cfg.encoding_dim)

        self.positional_encoding = nn.Parameter(torch.randn(1, max_seq_len, cfg.positional_encoding_dim))

        self.segmentation_mlp_encoder = StandardMLP(input_dim=cfg.encoding_dim, **cfg.segmentation_mlp_encoder_params, output_dim=cfg.segmentation_transformer_dim)
        segmentation_transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=cfg.segmentation_transformer_dim, **cfg.segmentation_transformer_encoder_layer_params)
        self.segmentation_transformer_encoder = nn.TransformerEncoder(segmentation_transformer_encoder_layer, **cfg.segmentation_transformer_encoder_params)
        self.segmentation_post = StandardMLP(input_dim=cfg.segmentation_transformer_dim, **cfg.segmentation_post_params, output_dim=2)

        self.compression_mlp_encoder = StandardMLP(input_dim=cfg.encoding_dim + cfg.positional_encoding_dim, **cfg.compression_mlp_encoder_params, output_dim=cfg.compression_transformer_dim)
        compression_transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=cfg.compression_transformer_dim, **cfg.compression_transformer_encoder_layer_params)
        self.compression_transformer_encoder = nn.TransformerEncoder(compression_transformer_encoder_layer, **cfg.compression_transformer_params)
        self.compression_transformer_nheads = cfg.compression_transformer_encoder_layer_params['nhead']
        self.abstract_rep_post = StandardMLP(input_dim=cfg.compression_transformer_dim, **cfg.abstract_rep_post_params, output_dim=cfg.abstract_rep_stoch_dim * 2)

        self.abstract_rep_mlp_encoder = StandardMLP(input_dim=cfg.abstract_rep_stoch_dim + cfg.positional_encoding_dim, **cfg.abstract_rep_mlp_encoder_params, output_dim=cfg.abstract_rep_transformer_dim)
        abstract_rep_transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=cfg.abstract_rep_transformer_dim, **cfg.abstract_rep_transformer_encoder_layer_params)
        self.abstract_rep_transformer_encoder = nn.TransformerEncoder(abstract_rep_transformer_encoder_layer, **cfg.abstract_rep_transformer_params)
        self.abstract_rep_transformer_nheads = cfg.abstract_rep_transformer_encoder_layer_params['nhead']
        self.abstract_rep_mlp_decoder = StandardMLP(input_dim=cfg.abstract_rep_transformer_dim, **cfg.abstract_rep_mlp_decoder_params, output_dim=cfg.abstract_rep_deter_dim)
        self.abstract_rep_prior = StandardMLP(input_dim=cfg.abstract_rep_deter_dim, **cfg.abstract_rep_prior_params, output_dim=cfg.abstract_rep_stoch_dim * 2)
        self.abstract_rep_dim = cfg.abstract_rep_stoch_dim + cfg.abstract_rep_deter_dim

        self.state_rep_post = StandardMLP(input_dim=cfg.encoding_dim + self.abstract_rep_dim, **cfg.state_rep_post_params, output_dim=cfg.state_rep_stoch_dim * 2)
        self.state_rep_context_encoder = StandardMLP(input_dim=self.abstract_rep_dim + cfg.positional_encoding_dim, **cfg.state_rep_context_encoder_params, output_dim=cfg.state_rep_transformer_dim)
        self.state_rep_mlp_encoder = StandardMLP(input_dim=cfg.state_rep_stoch_dim  + self.abstract_rep_dim + cfg.positional_encoding_dim, **cfg.state_rep_mlp_encoder_params, output_dim=cfg.state_rep_transformer_dim)
        state_rep_transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=cfg.state_rep_transformer_dim, **cfg.state_rep_transformer_encoder_layer_params)
        self.state_rep_transformer_encoder = nn.TransformerEncoder(state_rep_transformer_encoder_layer, **cfg.state_rep_transformer_params)
        self.state_rep_transformer_nheads = cfg.state_rep_transformer_encoder_layer_params['nhead']
        self.state_rep_mlp_decoder = StandardMLP(input_dim=cfg.state_rep_transformer_dim, **cfg.state_rep_mlp_decoder_params, output_dim=cfg.state_rep_deter_dim)
        self.state_rep_prior = StandardMLP(input_dim=cfg.state_rep_deter_dim + self.abstract_rep_dim, **cfg.state_rep_prior_params, output_dim=cfg.state_rep_stoch_dim * 2)
        self.state_rep_dim = cfg.state_rep_stoch_dim + cfg.state_rep_deter_dim

        self.segmentation_prior = StandardMLP(input_dim=self.state_rep_dim + self.abstract_rep_dim, **cfg.segmentation_prior_params, output_dim=2)

        self.decoder = StandardMLP(input_dim=self.state_rep_dim, **cfg.decoder_params, output_dim=data_dim)

    def forward(self, traj):
        # traj is a tensor of shape (batch_size, seq_len, data_dim)
        assert traj.shape[1] <= self.max_seq_len, "Trajectories must be less than length {}".format(self.seq_len)
        batch_size = traj.shape[0]
        encodings = self.encoder(traj)
        broadcast_positional_encoding = self.positional_encoding[:traj.shape[1]].expand(batch_size, -1, -1)

        segmentation_encodings = self.segmentation_mlp_encoder(encodings)
        transformed_segmentation_encodings = self.segmentation_transformer_encoder(segmentation_encodings)
        segmentation_logits = self.segmentation_post(transformed_segmentation_encodings)[:, 1:, :]
        segmentation_samples = nn.functional.gumbel_softmax(segmentation_logits, tau=self.temperature, hard=self.sample, dim=-1)[..., 1]
        segmentation_samples = torch.cat([torch.ones_like(segmentation_samples[:, :1]), segmentation_samples], dim=1)
        segmentation_attention_mask, causal_segmentation_attention_mask, abstract_causal_segmentation_attention_mask = self.get_segmentation_attention_masks(segmentation_samples)

        compression_encodings = self.compression_mlp_encoder(torch.cat([encodings, broadcast_positional_encoding], dim=-1))
        transformed_compression_encodings = prepend_null_token_transformer_encoder_pass(compression_encodings, segmentation_attention_mask, self.compression_transformer_encoder, nheads=self.compression_transformer_nheads)
        abstract_rep_post_params = self.abstract_rep_post(transformed_compression_encodings)
        abstract_rep_post_means, abstract_rep_post_stds = abstract_rep_post_params[..., :self.cfg.abstract_rep_stoch_dim], nn.functional.softplus(abstract_rep_post_params[..., self.cfg.abstract_rep_stoch_dim:])
        abstract_rep_stoch_samples = self.reparameterize(abstract_rep_post_means, abstract_rep_post_stds)

        abstract_rep_encodings = self.abstract_rep_mlp_encoder(torch.cat([abstract_rep_stoch_samples, broadcast_positional_encoding], dim=-1) * segmentation_samples.unsqueeze(-1))
        transformed_abstract_rep_encodings = prepend_null_token_transformer_encoder_pass(abstract_rep_encodings, abstract_causal_segmentation_attention_mask, self.abstract_rep_transformer_encoder, nheads=self.abstract_rep_transformer_nheads)
        abstract_rep_deter = self.abstract_rep_mlp_decoder(transformed_abstract_rep_encodings)
        abstract_rep_prior_params = self.abstract_rep_prior(abstract_rep_deter)
        abstract_rep_prior_means, abstract_rep_prior_stds = abstract_rep_prior_params[..., :self.cfg.abstract_rep_stoch_dim], nn.functional.softplus(abstract_rep_prior_params[..., self.cfg.abstract_rep_stoch_dim:])
        abstract_rep = torch.cat([abstract_rep_stoch_samples, abstract_rep_deter], dim=-1)

        state_rep_post_params = self.state_rep_post(torch.cat([encodings, abstract_rep], dim=-1))
        state_rep_post_means, state_rep_post_stds = state_rep_post_params[..., :self.cfg.state_rep_stoch_dim], nn.functional.softplus(state_rep_post_params[..., self.cfg.state_rep_stoch_dim:])
        state_rep_stoch_samples = self.reparameterize(state_rep_post_means, state_rep_post_stds)

        state_rep_encodings = self.state_rep_mlp_encoder(torch.cat([state_rep_stoch_samples, abstract_rep, broadcast_positional_encoding], dim=-1))
        transformed_state_rep_encodings = prepend_null_token_transformer_encoder_pass(state_rep_encodings, causal_segmentation_attention_mask, self.state_rep_transformer_encoder, nheads=self.state_rep_transformer_nheads)
        state_rep_deter = self.state_rep_mlp_decoder(transformed_state_rep_encodings)
        state_rep_prior_params = self.state_rep_prior(torch.cat([state_rep_deter, abstract_rep], dim=-1))
        state_rep_prior_means, state_rep_prior_stds = state_rep_prior_params[..., :self.cfg.state_rep_stoch_dim], nn.functional.softplus(state_rep_prior_params[..., self.cfg.state_rep_stoch_dim:])
        state_rep = torch.cat([state_rep_stoch_samples, state_rep_deter], dim=-1)

        reconstructed_traj = self.decoder(state_rep)

        return reconstructed_traj, dict(
            segmentation_logits=segmentation_logits,
            segmentation_samples=segmentation_samples,
            abstract_rep_post_means=abstract_rep_post_means,
            abstract_rep_post_stds=abstract_rep_post_stds,
            abstract_rep_prior_means=abstract_rep_prior_means,
            abstract_rep_prior_stds=abstract_rep_prior_stds,
            abstract_rep=abstract_rep,
            state_rep_post_means=state_rep_post_means,
            state_rep_post_stds=state_rep_post_stds,
            state_rep_prior_means=state_rep_prior_means,
            state_rep_prior_stds=state_rep_prior_stds,
            state_rep=state_rep,
        )

    def get_loss(self, traj):
        reconstructed_traj, info = self.forward(traj)
        reconstruction_loss = nn.functional.mse_loss(traj, reconstructed_traj)
        segmentation_samples = info['segmentation_samples'].unsqueeze(-1)
        time_loss = torch.mean(segmentation_samples[:, 1:])

        abstract_rep_post_means, abstract_rep_post_stds = info['abstract_rep_post_means'], info['abstract_rep_post_stds']
        abstract_rep_post_dist = torch.distributions.Normal(abstract_rep_post_means, abstract_rep_post_stds)
        abstract_rep_prior_means, abstract_rep_prior_stds = info['abstract_rep_prior_means'], info['abstract_rep_prior_stds']
        abstract_rep_prior_dist = torch.distributions.Normal(abstract_rep_prior_means, abstract_rep_prior_stds)

        # TODO: KL Balancing, don't regularize posterior to bad prior
        abstract_rep_kl_loss = torch.sum(torch.distributions.kl_divergence(abstract_rep_post_dist, abstract_rep_prior_dist) * segmentation_samples)

        state_rep_post_means, state_rep_post_stds = info['state_rep_post_means'], info['state_rep_post_stds']
        state_rep_post_dist = torch.distributions.Normal(state_rep_post_means, state_rep_post_stds)
        state_rep_prior_means, state_rep_prior_stds = info['state_rep_prior_means'], info['state_rep_prior_stds']
        state_rep_prior_dist = torch.distributions.Normal(state_rep_prior_means, state_rep_prior_stds)

        # TODO: KL Balancing, don't regularize posterior to bad prior
        state_rep_kl_loss = torch.sum(torch.distributions.kl_divergence(state_rep_post_dist, state_rep_prior_dist) * segmentation_samples)

        model_loss = self.cfg.reconstruction_loss_weight * reconstruction_loss + \
            self.cfg.time_loss_weight * time_loss + \
            self.cfg.abstract_transition_kl_weight * abstract_rep_kl_loss + \
            self.cfg.state_transition_kl_weight * state_rep_kl_loss

        metrics = dict(
            loss=model_loss.item(),
            reconstruction_loss=reconstruction_loss.item(),
            time_loss=time_loss.item(),
            abstract_transition_kl_loss=abstract_rep_kl_loss.item(),
            state_transition_kl_loss=state_rep_kl_loss.item(),
        )

        return model_loss, metrics, info

    def reparameterize(self, means, stds):
        eps = torch.randn_like(stds)
        return means + eps * stds

    def get_segmentation_attention_masks(self, delta_t):
        device = delta_t.device
        batch_size = delta_t.shape[0]
        seq_len = delta_t.shape[1]
        assert seq_len <= self.max_seq_len
        padding = torch.ones(batch_size, seq_len, device=device)
        delta_t = torch.cat([padding, delta_t, padding], dim=1)
        attention_weights = deque([torch.ones(batch_size, seq_len, device=device)])

        forward_elapsed_t = torch.zeros(batch_size, seq_len, device=device)
        backward_elapsed_t = torch.zeros(batch_size, seq_len, device=device)

        base_indices = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, seq_len) + seq_len
        for i in range(seq_len):
            forward_indices = base_indices + i + 1
            backward_indices = base_indices - i

            forward_elapsed_t = forward_elapsed_t + torch.gather(delta_t, 1, forward_indices)
            backward_elapsed_t = backward_elapsed_t + torch.gather(delta_t, 1, backward_indices)

            attention_weights.append(torch.maximum(1 - forward_elapsed_t, torch.zeros_like(forward_elapsed_t)))
            attention_weights.appendleft(torch.maximum(1 - backward_elapsed_t, torch.zeros_like(backward_elapsed_t)))

        attention_weights = torch.stack(list(attention_weights), dim=-1)
        seq_indices = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).unsqueeze(1) + (seq_len - torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, seq_len)).unsqueeze(-1)
        attention_weights = torch.gather(attention_weights, 2, seq_indices)
        segmentation_attention_mask = attention_weights.unsqueeze(1).expand(batch_size, self.compression_transformer_nheads, seq_len, seq_len)
        segmentation_attention_mask = segmentation_attention_mask.reshape(batch_size * self.compression_transformer_nheads, seq_len, seq_len)

        all_indices = seq_indices - seq_len
        causal_attention_weights = torch.where(torch.zeros(1, device=device) > all_indices, attention_weights, torch.zeros_like(attention_weights))

        causal_segmentation_attention_mask = causal_attention_weights.unsqueeze(1).expand(batch_size, self.state_rep_transformer_nheads, seq_len, seq_len)
        causal_segmentation_attention_mask = causal_segmentation_attention_mask.reshape(batch_size * self.state_rep_transformer_nheads, seq_len, seq_len)

        abstract_causal_attention_weights = torch.where(torch.zeros(1, device=device) > all_indices, torch.ones_like(attention_weights) - attention_weights, torch.zeros_like(attention_weights))

        abstract_causal_segmentation_attention_mask = abstract_causal_attention_weights.unsqueeze(1).expand(batch_size, self.abstract_rep_transformer_nheads, seq_len, seq_len)
        abstract_causal_segmentation_attention_mask = abstract_causal_segmentation_attention_mask.reshape(batch_size * self.abstract_rep_transformer_nheads, seq_len, seq_len)

        return torch.log(segmentation_attention_mask), torch.log(causal_segmentation_attention_mask), torch.log(abstract_causal_segmentation_attention_mask)

    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_time_loss_weight(self, time_loss_weight):
        self.time_loss_weight = time_loss_weight

    def hard_sample(self):
        self.sample = True

    def soft_sample(self):
        self.sample = False

    def train(self, mode=True):
        if mode:
            self.soft_sample()
        else:
            self.hard_sample()
        super().train(mode)

    def eval(self):
        self.hard_sample()
        super().eval()

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

def prepend_null_token_transformer_encoder_pass(transformer_input, transformer_mask, transformer, nheads, null_token=None, null_mask_value=0):
    # Expecting batch first input, so (batch_size, seq_len, input_dim)
    batch_size, seq_len, _ = transformer_input.shape
    device = transformer_input.device
    if null_token is None:
        null_token = torch.zeros_like(transformer_input[:, :1, :])
    null_vector_to_others_mask = torch.zeros(batch_size * nheads, 1, seq_len, device=device)
    all_vectors_to_null_mask = torch.ones(batch_size * nheads, seq_len + 1, 1, device=device) * null_mask_value

    transformer_input = torch.cat([null_token, transformer_input], dim=1)
    transformer_mask = torch.cat([torch.cat([null_vector_to_others_mask, transformer_mask], dim=1), all_vectors_to_null_mask], dim=2)
    transformer_output = transformer(transformer_input, transformer_mask)
    return transformer_output[:, 1:, :]

def test_temporal_attention():
    l = 10
    model = PrototypeModel(seq_len=10, max_subseq_len=5)
    trajs = torch.randn(2, l, 1)
    model.get_loss(trajs)

def multihead_attention_mask_shape_test():
    batch_size = 3
    embedding_dim = 4
    num_heads = 2
    sequence_len = 5
    mha = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
    q = torch.randn(batch_size, sequence_len, embedding_dim)
    k = torch.randn(batch_size, sequence_len, embedding_dim)
    v = torch.randn(batch_size, sequence_len, embedding_dim)
    mask = torch.where(torch.rand(batch_size, sequence_len, sequence_len) > 0.5, -torch.inf * torch.ones(1), torch.zeros(1))
    mask = mask.unsqueeze(1).repeat(1, num_heads, 1, 1).reshape(batch_size * num_heads, sequence_len, sequence_len)
    print("Mask: ")
    print(mask)
    print(mask.shape)
    print("Reshaped mask: ")
    print(mask.reshape(batch_size, num_heads, sequence_len, sequence_len))
    print(mask.reshape(batch_size, num_heads, sequence_len, sequence_len).shape)
    _, weights = mha(q, k, v, attn_mask=mask)
    print("Weights: ")
    print(weights)
    print(weights.shape)

def test_full_prototype_forward():
    from hydra import initialize, compose

    batch_size = 2
    seq_len = 5
    data_dim = 1

    with initialize(version_base="1.3", config_path="cfgs/model/"):
        cfg = compose(config_name="full_prototype_v1")
        model = FullPrototypeModel(cfg, data_dim=data_dim, max_seq_len=seq_len)

        traj = torch.randn(batch_size, seq_len, data_dim)
        _, metrics, _ = model.get_loss(traj)
        print(metrics)

if __name__ == '__main__':
    # test_temporal_attention()
    # multihead_attention_mask_shape_test()
    test_full_prototype_forward()
    pass
