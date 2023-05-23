from collections import deque
import copy

import numpy as np
import torch
import torch.nn as nn

import concrete
import rpr

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
        self.state_kl_weight = cfg.state_transition_kl_weight

        self.encoder = StandardMLP(input_dim=data_dim, **cfg.encoder_params, output_dim=cfg.encoding_dim)

        self.segmentation_mlp_encoder = StandardMLP(input_dim=cfg.encoding_dim, **cfg.segmentation_mlp_encoder_params, output_dim=cfg.segmentation_transformer_dim)
        segmentation_transformer_encoder_layer = rpr.TransformerEncoderLayerRPR(d_model=cfg.segmentation_transformer_dim, **cfg.segmentation_transformer_encoder_layer_params, er_len=max_seq_len)
        self.segmentation_transformer_encoder = rpr.TransformerEncoderRPR(segmentation_transformer_encoder_layer, **cfg.segmentation_transformer_encoder_params)
        self.segmentation_gru = nn.GRUCell(cfg.segmentation_transformer_dim + 1, cfg.segmentation_transformer_dim)
        self.segmentation_post = StandardMLP(input_dim=cfg.segmentation_transformer_dim, **cfg.segmentation_post_params, output_dim=1)

        self.compression_mlp_encoder = StandardMLP(input_dim=cfg.encoding_dim + cfg.segmentation_transformer_dim, **cfg.compression_mlp_encoder_params, output_dim=cfg.temporal_attention_dim)
        # self.compression_mlp_encoder = StandardMLP(input_dim=cfg.encoding_dim + 1, **cfg.compression_mlp_encoder_params, output_dim=cfg.compression_transformer_dim)
        # self.compression_transformer_encoder_layer = rpr.TransformerEncoderLayerRPR(d_model=cfg.compression_transformer_dim, **cfg.compression_transformer_encoder_layer_params, er_len=max_seq_len)
        # self.compression_transfomer = rpr.TransformerEncoderRPR(self.compression_transformer_encoder_layer, **cfg.compression_transformer_params)
        # self.compression_mlp_decoder = StandardMLP(input_dim=cfg.compression_transformer_dim, **cfg.compression_mlp_decoder_params, output_dim=cfg.temporal_attention_dim)
        self.abstract_rep_post = StandardMLP(input_dim=cfg.temporal_attention_dim, **cfg.abstract_rep_post_params, output_dim=cfg.abstract_rep_stoch_dim * 2)

        self.abstract_rep_mlp_encoder = StandardMLP(input_dim=cfg.abstract_rep_stoch_dim, **cfg.abstract_rep_mlp_encoder_params, output_dim=cfg.abstract_rep_transformer_dim)
        abstract_rep_transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=cfg.abstract_rep_transformer_dim, **cfg.abstract_rep_transformer_encoder_layer_params)
        self.abstract_rep_transformer_encoder = nn.TransformerEncoder(abstract_rep_transformer_encoder_layer, **cfg.abstract_rep_transformer_params)
        self.abstract_rep_transformer_nheads = cfg.abstract_rep_transformer_encoder_layer_params['nhead']
        self.abstract_rep_mlp_decoder = StandardMLP(input_dim=cfg.abstract_rep_transformer_dim, **cfg.abstract_rep_mlp_decoder_params, output_dim=cfg.abstract_rep_deter_dim)
        self.abstract_rep_prior = StandardMLP(input_dim=cfg.abstract_rep_deter_dim, **cfg.abstract_rep_prior_params, output_dim=cfg.abstract_rep_stoch_dim * 2)
        self.abstract_rep_dim = cfg.abstract_rep_stoch_dim + cfg.abstract_rep_deter_dim
        # self.abstract_rep_prior = StandardMLP(input_dim=cfg.abstract_rep_stoch_dim, **cfg.abstract_rep_prior_params, output_dim=cfg.abstract_rep_stoch_dim * 2)
        # self.abstract_rep_dim = cfg.abstract_rep_stoch_dim

        self.state_rep_post = StandardMLP(input_dim=cfg.encoding_dim + self.abstract_rep_dim, **cfg.state_rep_post_params, output_dim=cfg.state_rep_stoch_dim * 2)
        self.state_rep_mlp_encoder = StandardMLP(input_dim=cfg.state_rep_stoch_dim  + self.abstract_rep_dim, **cfg.state_rep_mlp_encoder_params, output_dim=cfg.state_rep_transformer_dim)
        state_rep_transformer_encoder_layer = rpr.TransformerEncoderLayerRPR(d_model=cfg.state_rep_transformer_dim, **cfg.state_rep_transformer_encoder_layer_params, er_len=max_seq_len)
        self.state_rep_transformer_encoder = rpr.TransformerEncoderRPR(state_rep_transformer_encoder_layer, **cfg.state_rep_transformer_params)
        self.state_rep_transformer_nheads = cfg.state_rep_transformer_encoder_layer_params['nhead']
        self.state_rep_mlp_decoder = StandardMLP(input_dim=cfg.state_rep_transformer_dim, **cfg.state_rep_mlp_decoder_params, output_dim=cfg.state_rep_deter_dim)
        self.state_rep_prior = StandardMLP(input_dim=cfg.state_rep_deter_dim + self.abstract_rep_dim, **cfg.state_rep_prior_params, output_dim=cfg.state_rep_stoch_dim * 2)
        self.state_rep_dim = cfg.state_rep_stoch_dim + cfg.state_rep_deter_dim

        self.segmentation_prior = StandardMLP(input_dim=self.state_rep_dim, **cfg.segmentation_prior_params, output_dim=1)

        self.decoder = StandardMLP(input_dim=self.state_rep_dim, **cfg.decoder_params, output_dim=data_dim)

    def forward(self, traj, abstract_sample_std_scalar=1.0, state_sample_std_scalar=1.0):
        # traj is a tensor of shape (batch_size, seq_len, data_dim)
        batch_size = traj.shape[0]
        seq_len = traj.shape[1]
        device = traj.device
        assert seq_len <= self.max_seq_len, "Trajectories must be less than length {}".format(self.seq_len)
        encodings = self.encoder(traj)
        broadcast_positional_encoding = self.positional_encoding_dropout(self.positional_encoding[:, :traj.shape[1]].expand(batch_size, -1, -1))

        segmentation_encodings = self.segmentation_mlp_encoder(encodings)
        transformed_segmentation_encodings = self.segmentation_transformer_encoder(torch.transpose(segmentation_encodings, 0, 1)).transpose(0, 1) # TRANSPOSE FOR RPR TRANSFORMER
        segmentation_post_logits = []
        segmentation_post_probs = [torch.ones(batch_size, 1, device=device, dtype=torch.float32)]
        segmentation_samples = [torch.ones(batch_size, 1, device=device, dtype=torch.float32)]
        y_samples = []
        gru_hidden = torch.zeros(batch_size, self.cfg.segmentation_transformer_dim, device=device, dtype=torch.float32)
        for i in range(1, seq_len):
            gru_hidden = self.segmentation_gru(torch.cat([segmentation_post_probs[-1], transformed_segmentation_encodings[:, i, :]], dim=-1), gru_hidden)
            segmentation_post_logit = self.segmentation_post(gru_hidden)
            segmentation_sample, y_sample = concrete.sample_binary_concrete(segmentation_post_logit, self.temperature, hard=self.sample)
            segmentation_post_probs.append(torch.sigmoid(segmentation_post_logit))
            segmentation_post_logits.append(segmentation_post_logit)
            if self.cfg['fix_segmentation_period'] is None:
                segmentation_samples.append(segmentation_sample)
            else:
                if i % self.cfg['fix_segmentation_period'] == 0:
                    segmentation_samples.append(torch.ones_like(segmentation_sample))
                else:
                    segmentation_samples.append(torch.zeros_like(segmentation_sample))
            y_samples.append(y_sample)

        segmentation_post_logits = torch.stack(segmentation_post_logits, dim=1)
        segmentation_samples = torch.stack(segmentation_samples, dim=1)
        y_samples = torch.stack(y_samples, dim=1)

        segmentation_samples = segmentation_samples.squeeze(-1)
        if segmentation_samples.requires_grad:
            segmentation_samples.register_hook(lambda grad: self.cfg.time_grad_scalar * grad)
        # segmentation_samples = torch.sigmoid(segmentation_logits)[..., 0]
        # segmentation_samples = torch.bernoulli(segmentation_samples) + segmentation_samples - segmentation_samples.detach()
        # segmentation_samples = torch.cat([torch.ones_like(segmentation_samples[:, :1]), segmentation_samples], dim=1)
        if segmentation_samples.requires_grad:
            segmentation_samples.retain_grad()
        segment_weights, _, causal_segmentation_attention_mask, abstract_causal_segmentation_attention_mask = self.get_segmentation_attention_masks_probabilistic(segmentation_samples)

        query_encodings = self.query_mlp_encoder(torch.cat([encodings, broadcast_positional_encoding], dim=-1))
        repeated_query_encodings = torch.cat([query_encodings] * seq_len, dim=1).reshape(batch_size, seq_len, seq_len, self.cfg.query_attention_dim)
        attended_query_encodings = torch.sum(segment_weights.unsqueeze(-1) * repeated_query_encodings, dim=2)
        abstract_rep_post_params = self.abstract_rep_post(attended_query_encodings)
        abstract_rep_post_means, abstract_rep_post_stds = abstract_rep_post_params[..., :self.cfg.abstract_rep_stoch_dim], nn.functional.softplus(abstract_rep_post_params[..., self.cfg.abstract_rep_stoch_dim:])
        abstract_rep_stoch_samples = self.reparameterize_segments(abstract_rep_post_means, abstract_rep_post_stds, segmentation_samples, std_scalar=abstract_sample_std_scalar)

        # abstract_rep_encodings = self.abstract_rep_mlp_encoder(torch.cat([abstract_rep_stoch_samples, broadcast_positional_encoding], dim=-1))
        # transformed_abstract_rep_encodings = self.abstract_rep_transformer_encoder(abstract_rep_stoch_samples, abstract_rep_encodings, mask=abstract_causal_segmentation_attention_mask)
        # abstract_rep_deter = self.abstract_rep_mlp_decoder(transformed_abstract_rep_encodings)
        # abstract_rep_prior_params = self.abstract_rep_prior(shift_forward(abstract_rep_deter, 1))
        # abstract_rep_prior_means, abstract_rep_prior_stds = abstract_rep_prior_params[..., :self.cfg.abstract_rep_stoch_dim], nn.functional.softplus(abstract_rep_prior_params[..., self.cfg.abstract_rep_stoch_dim:])
        # abstract_rep = torch.cat([abstract_rep_stoch_samples, abstract_rep_deter], dim=-1)
        abstract_rep_prior_params = self.abstract_rep_prior(shift_forward(abstract_rep_stoch_samples, 1))
        abstract_rep_prior_means, abstract_rep_prior_stds = abstract_rep_prior_params[..., :self.cfg.abstract_rep_stoch_dim], nn.functional.softplus(abstract_rep_prior_params[..., self.cfg.abstract_rep_stoch_dim:])
        abstract_rep = abstract_rep_stoch_samples

        state_rep_post_params = self.state_rep_post(torch.cat([encodings, abstract_rep], dim=-1))
        state_rep_post_means, state_rep_post_stds = state_rep_post_params[..., :self.cfg.state_rep_stoch_dim], nn.functional.softplus(state_rep_post_params[..., self.cfg.state_rep_stoch_dim:])
        state_rep_stoch_samples = self.reparameterize(state_rep_post_means, state_rep_post_stds, std_scalar=state_sample_std_scalar)

        state_rep_encodings = self.state_rep_mlp_encoder(torch.cat([state_rep_stoch_samples, abstract_rep], dim=-1))
        transformed_state_rep_encodings = self.state_rep_transformer_encoder(torch.transpose(state_rep_encodings, 0, 1), mask=causal_segmentation_attention_mask).transpose(0, 1) # TRANSPOSE FOR RPR TRANSFORMER
        state_rep_deter = self.state_rep_mlp_decoder(transformed_state_rep_encodings)
        state_rep_prior_params = self.state_rep_prior(torch.cat([shift_forward(state_rep_deter, 1) * (1 - segmentation_samples).unsqueeze(-1), abstract_rep], dim=-1))
        state_rep_prior_means, state_rep_prior_stds = state_rep_prior_params[..., :self.cfg.state_rep_stoch_dim], nn.functional.softplus(state_rep_prior_params[..., self.cfg.state_rep_stoch_dim:])
        state_rep = torch.cat([state_rep_stoch_samples, state_rep_deter], dim=-1)

        # state_rep = torch.zeros_like(state_rep)
        decoder_input = torch.cat([state_rep, abstract_rep], dim=-1)
        segmentation_prior_logits = self.segmentation_prior(decoder_input)[:, :-1]
        reconstructed_traj = self.decoder(decoder_input)

        return reconstructed_traj, dict(
            segment_weights=segment_weights,
            segmentation_post_logits=segmentation_post_logits,
            segmentation_samples=segmentation_samples,
            y_samples=y_samples,
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
            segmentation_prior_logits=segmentation_prior_logits,
        )

    def get_loss(self, traj):
        reconstructed_traj, info = self.forward(traj)
        info['reconstructed_traj'] = reconstructed_traj
        info['ground_truth_traj'] = traj
        reconstruction_loss = nn.functional.mse_loss(traj, reconstructed_traj)
        segmentation_samples = info['segmentation_samples']
        average_compression = 1 / torch.mean(segmentation_samples[:, 1:])

        abstract_rep_post_means, abstract_rep_post_stds = info['abstract_rep_post_means'], info['abstract_rep_post_stds']
        abstract_rep_prior_means, abstract_rep_prior_stds = info['abstract_rep_prior_means'], info['abstract_rep_prior_stds']
        # abstract_rep_prior_means, abstract_rep_prior_stds = torch.zeros_like(abstract_rep_post_means), torch.ones_like(abstract_rep_post_stds)

        # TODO: Don't include time loss factor into KL loss, keep them factorized
        # abstract_rep_kl_loss = torch.mean((self.kl_balance_gaussian(abstract_rep_prior_means, abstract_rep_prior_stds, abstract_rep_post_means, abstract_rep_post_stds, self.cfg.abstract_kl_balance)) * segmentation_samples)
        abs_kl = self.kl_balance_gaussian(abstract_rep_prior_means, abstract_rep_prior_stds, abstract_rep_post_means, abstract_rep_post_stds, self.cfg.abstract_kl_balance)
        abstract_rep_kl_loss = torch.mean(torch.sum(abs_kl * segmentation_samples.detach(), dim=1) / torch.sum(segmentation_samples, dim=1))

        state_rep_post_means, state_rep_post_stds = info['state_rep_post_means'], info['state_rep_post_stds']
        state_rep_prior_means, state_rep_prior_stds = info['state_rep_prior_means'], info['state_rep_prior_stds']

        state_kl = self.kl_balance_gaussian(state_rep_prior_means, state_rep_prior_stds, state_rep_post_means, state_rep_post_stds, self.cfg.state_kl_balance)
        state_rep_kl_loss = torch.mean(state_kl)
        info['state_kl'] = state_kl

        segmentation_post_logits = info['segmentation_post_logits']
        segmentation_prior_logits = info['segmentation_prior_logits']
        segmentation_loss = torch.sigmoid(segmentation_post_logits).mean()
        temp = torch.tensor(self.temperature, device=segmentation_post_logits.device)
        segmentation_kl_loss = torch.mean(concrete.y_kl_divergence(info['y_samples'], segmentation_prior_logits, temp, segmentation_post_logits, temp, kl_balance=self.cfg.segmentation_kl_balance))

        model_loss = self.cfg.reconstruction_loss_weight * reconstruction_loss + \
            self.time_loss_weight * segmentation_loss + \
            self.cfg.abstract_transition_kl_weight * abstract_rep_kl_loss + \
            self.state_kl_weight * state_rep_kl_loss + \
            self.cfg.segmentation_kl_weight * segmentation_kl_loss

        metrics = dict(
            loss=model_loss.item(),
            reconstruction_loss=reconstruction_loss.item(),
            segmentation_loss=segmentation_loss.item(),
            average_compression=torch.minimum(average_compression, torch.tensor(self.max_seq_len, device=average_compression.device)).item(),
            abstract_transition_kl_loss=abstract_rep_kl_loss.item(),
            state_transition_kl_loss=state_rep_kl_loss.item(),
            segmentation_kl_loss=segmentation_kl_loss.item(),
        )

        return model_loss, metrics, info

    def generate(self, batch_size, abstract_sample_std_scalar=1, state_sample_std_scalar=1, generation_length=None, given_segmentations=None, given_abstract_stoch=None, given_state_stoch=None, initial_stoch=None):
        device = self.positional_encoding.device
        if generation_length is None:
            generation_length = self.max_seq_len

        broadcast_positional_encoding = self.positional_encoding_dropout(self.positional_encoding[:, :generation_length].expand(batch_size, -1, -1))

        segmentation_probs = torch.zeros(batch_size, generation_length, device=device, dtype=torch.float32)
        segmentations = torch.zeros(batch_size, generation_length, device=device, dtype=torch.float32)
        segmentation_probs[:, 0] = 1
        segmentations[:, 0] = 1
        if given_segmentations is not None:
            segmentations = given_segmentations

        # abstract_rep = torch.zeros(batch_size, generation_length, self.abstract_rep_dim, device=device, dtype=torch.float32)
        abstract_rep = torch.zeros(batch_size, generation_length, self.cfg.abstract_rep_stoch_dim, device=device, dtype=torch.float32)
        if given_abstract_stoch is not None:
            abstract_rep[..., :self.cfg.abstract_rep_stoch_dim] = given_abstract_stoch

        state_rep = torch.zeros(batch_size, generation_length, self.state_rep_dim, device=device, dtype=torch.float32)
        if given_state_stoch is not None:
            state_rep[..., :self.cfg.state_rep_stoch_dim] = given_state_stoch

        generated_traj = torch.zeros(batch_size, generation_length, self.data_dim, device=device, dtype=torch.float32)

        abstract_stoch_means = torch.zeros(batch_size, generation_length, self.cfg.abstract_rep_stoch_dim, device=device, dtype=torch.float32)
        state_stoch_means = torch.zeros(batch_size, generation_length, self.cfg.state_rep_stoch_dim, device=device, dtype=torch.float32)

        abstract_eps = torch.randn(batch_size, generation_length, self.cfg.abstract_rep_stoch_dim, device=device, dtype=torch.float32)
        abstract_seg_eps = torch.zeros_like(abstract_eps)

        for i in range(generation_length):
            # abstract_rep_prior_params = self.abstract_rep_prior(shift_forward(abstract_rep[:, :i + 1, -self.cfg.abstract_rep_deter_dim:], 1)[:, -1:])
            abstract_rep_prior_params = self.abstract_rep_prior(shift_forward(abstract_rep[:, :i + 1], 1)[:, -1:])
            abstract_rep_prior_means, abstract_rep_prior_stds = abstract_rep_prior_params[..., :self.cfg.abstract_rep_stoch_dim], nn.functional.softplus(abstract_rep_prior_params[..., self.cfg.abstract_rep_stoch_dim:])
            segment = segmentations[:, i:i + 1].unsqueeze(-1)
            if given_abstract_stoch is None:
                if i == 0:
                    abstract_seg_eps[:, i:i + 1] = abstract_eps[:, i:i + 1]
                    if initial_stoch is None:
                        abstract_rep_stoch_samples = abstract_rep_prior_means + abstract_rep_prior_stds * abstract_seg_eps[:, i:i + 1] * abstract_sample_std_scalar
                    else:
                        abstract_rep_stoch_samples = initial_stoch
                else:
                    abstract_seg_eps[:, i:i + 1] = (1 - segment) * abstract_seg_eps[:, i - 1:i] + segment * abstract_eps[:, i:i + 1]
                    abstract_rep_stoch_samples = (1 - segment) * abstract_rep[:, i - 1:i, :self.cfg.abstract_rep_stoch_dim] + segment * (abstract_rep_prior_means + abstract_rep_prior_stds * abstract_seg_eps[:, i:i + 1] * abstract_sample_std_scalar)
                abstract_rep[:, i:i + 1, :self.cfg.abstract_rep_stoch_dim] = abstract_rep_stoch_samples
            abstract_stoch_means[:, i:i + 1] = abstract_rep_prior_means

            segmentation_samples = segmentations[:, :i + 1]
            _, _, causal_segmentation_attention_mask, abstract_causal_segmentation_attention_mask = self.get_segmentation_attention_masks_probabilistic(segmentation_samples)

            # abstract_stoch_hist = abstract_rep[:, :i + 1, :self.cfg.abstract_rep_stoch_dim]
            # abstract_rep_encodings = self.abstract_rep_mlp_encoder(torch.cat([abstract_stoch_hist, broadcast_positional_encoding[:, :i + 1]], dim=-1))
            # transformed_abstract_rep_encodings = self.abstract_rep_transformer_encoder(abstract_stoch_hist, abstract_rep_encodings, mask=abstract_causal_segmentation_attention_mask)
            # abstract_rep_deter = self.abstract_rep_mlp_decoder(transformed_abstract_rep_encodings)
            # abstract_rep[:, i:i + 1, -self.cfg.abstract_rep_deter_dim:] = abstract_rep_deter[:, -1:]

            state_rep_prior_params = self.state_rep_prior(torch.cat([(shift_forward(state_rep[:, :i + 1, -self.cfg.state_rep_deter_dim:], 1))[:, -1:] * (1 - segmentation_samples[:, -1:]).unsqueeze(-1), abstract_rep[:, i:i + 1]], dim=-1))
            state_rep_prior_means, state_rep_prior_stds = state_rep_prior_params[..., :self.cfg.state_rep_stoch_dim], nn.functional.softplus(state_rep_prior_params[..., self.cfg.state_rep_stoch_dim:])
            if given_state_stoch is None:
                state_rep_stoch = state_rep_prior_means + state_rep_prior_stds * torch.randn_like(state_rep_prior_means) * state_sample_std_scalar
                state_rep[:, i:i + 1, :self.cfg.state_rep_stoch_dim] = state_rep_stoch
            state_stoch_means[:, i:i + 1] = state_rep_prior_means

            state_stoch_hist = state_rep[:, :i + 1, :self.cfg.state_rep_stoch_dim]
            abstract_rep_hist = abstract_rep[:, :i + 1]
            state_rep_encodings = self.state_rep_mlp_encoder(torch.cat([state_stoch_hist, abstract_rep_hist], dim=-1))
            transformed_state_rep_encodings = self.state_rep_transformer_encoder(torch.transpose(state_rep_encodings, 0, 1), mask=causal_segmentation_attention_mask).transpose(0, 1)
            state_rep_deter = self.state_rep_mlp_decoder(transformed_state_rep_encodings)
            state_rep[:, i:i + 1, -self.cfg.state_rep_deter_dim:] = state_rep_deter[:, -1:]

            decoder_input = torch.cat([state_rep[:, i:i + 1], abstract_rep[:, i:i + 1]], dim=-1)
            generated_traj[:, i:i + 1] = self.decoder(decoder_input)

            if i < generation_length - 1:
                segmentation_prior_logits = self.segmentation_prior(decoder_input)
                segmentation_samples = torch.distributions.Bernoulli(logits=segmentation_prior_logits).sample().squeeze(-1)
                segmentation_probs[:, i + 1:i + 2] = torch.sigmoid(segmentation_prior_logits).squeeze(-1)
                if given_segmentations is None:
                    if self.cfg['fix_segmentation_period'] is None:
                        segmentations[:, i + 1:i + 2] = segmentation_samples
                    else:
                        if (i + 1) % self.cfg['fix_segmentation_period'] == 0:
                            segmentations[:, i + 1:i + 2] = torch.ones_like(segmentation_samples)
                        else:
                            segmentations[:, i + 1:i + 2] = torch.zeros_like(segmentation_samples)

        return generated_traj, dict(
            abstract_rep=abstract_rep,
            state_rep=state_rep,
            segmentation_samples=segmentations,
            segmentation_probs=segmentation_probs,
            abstract_stoch_means=abstract_stoch_means,
            state_stoch_means=state_stoch_means,
        )

    def get_dist_gaussian(self, means, stds):
        return torch.distributions.Independent(torch.distributions.Normal(means, stds), 1)

    def kl_balance_gaussian(self, prior_means, prior_stds, post_means, post_stds, kl_balance_ratio):
        kl_prior = torch.distributions.kl_divergence(self.get_dist_gaussian(post_means.detach(), post_stds.detach()), self.get_dist_gaussian(prior_means, prior_stds))
        kl_post = torch.distributions.kl_divergence(self.get_dist_gaussian(post_means, post_stds), self.get_dist_gaussian(prior_means.detach(), prior_stds.detach()))
        kl_balanced = kl_balance_ratio * kl_prior + (1 - kl_balance_ratio) * kl_post
        return kl_balanced

    def get_dist_bernoulli(self, logits):
        return torch.distributions.Independent(torch.distributions.Bernoulli(logits=logits), 1)

    def kl_balance_bernoulli(self, prior_logits, post_logits, kl_balance_ratio):
        kl_prior = torch.distributions.kl_divergence(self.get_dist_bernoulli(post_logits.detach()), self.get_dist_bernoulli(prior_logits))
        kl_post = torch.distributions.kl_divergence(self.get_dist_bernoulli(post_logits), self.get_dist_bernoulli(prior_logits.detach()))
        kl_balanced = kl_balance_ratio * kl_prior + (1 - kl_balance_ratio) * kl_post
        return kl_balanced

    def reparameterize(self, means, stds, std_scalar=1.0):
        eps = torch.randn_like(stds)
        return means + eps * stds * std_scalar

    def reparameterize_segments(self, means, stds, segmentations, std_scalar=1.0):
        segmentations = segmentations.unsqueeze(-1)
        eps = torch.randn_like(stds)
        seg_eps = [eps[:, 0]]
        for i in range(1, eps.shape[1]):
            seg_eps.append((1 - segmentations[:, i]) * seg_eps[-1] + segmentations[:, i] * eps[:, i])
        seg_eps = torch.stack(seg_eps, dim=1)
        return means + seg_eps * stds * std_scalar

    def get_segmentation_attention_masks_probabilistic(self, segmentation_probs):
        device = segmentation_probs.device
        batch_size = segmentation_probs.shape[0]
        seq_len = segmentation_probs.shape[1]
        assert seq_len <= self.max_seq_len
        padding = torch.ones(batch_size, seq_len, device=device)
        segmentation_probs_padded = torch.cat([padding, segmentation_probs, padding], dim=1)
        attention_weights = deque([torch.ones(batch_size, seq_len, device=device)])

        forward_p_same_segment = torch.ones(batch_size, seq_len, device=device)
        backward_p_same_segment = torch.ones(batch_size, seq_len, device=device)

        no_segmentation_probs_padded = 1 - segmentation_probs_padded

        base_indices = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, seq_len) + seq_len
        for i in range(seq_len):
            forward_indices = base_indices + i + 1
            backward_indices = base_indices - i

            forward_p_same_segment = forward_p_same_segment * torch.gather(no_segmentation_probs_padded, 1, forward_indices)
            backward_p_same_segment = backward_p_same_segment * torch.gather(no_segmentation_probs_padded, 1, backward_indices)

            attention_weights.append(forward_p_same_segment)
            attention_weights.appendleft(backward_p_same_segment)

        attention_weights = torch.stack(list(attention_weights), dim=-1)
        seq_indices = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).unsqueeze(1) + (seq_len - torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, seq_len)).unsqueeze(-1)
        attention_weights = torch.gather(attention_weights, 2, seq_indices)

        all_indices = seq_indices - seq_len
        causal_attention_weights = torch.where(torch.zeros(1, device=device) >= all_indices, attention_weights, torch.zeros_like(attention_weights))

        causal_segmentation_attention_mask = causal_attention_weights.unsqueeze(1).expand(batch_size, self.state_rep_transformer_nheads, seq_len, seq_len)
        causal_segmentation_attention_mask = causal_segmentation_attention_mask.reshape(batch_size * self.state_rep_transformer_nheads, seq_len, seq_len)

        abstract_causal_attention_weights = torch.where(torch.zeros(1, device=device) >= all_indices, torch.cat([segmentation_probs] * seq_len, dim=1).reshape(batch_size, seq_len, seq_len), torch.zeros_like(attention_weights))

        abstract_causal_segmentation_attention_mask = abstract_causal_attention_weights.unsqueeze(1).expand(batch_size, self.abstract_rep_transformer_nheads, seq_len, seq_len)
        abstract_causal_segmentation_attention_mask = abstract_causal_segmentation_attention_mask.reshape(batch_size * self.abstract_rep_transformer_nheads, seq_len, seq_len)
        if causal_segmentation_attention_mask.requires_grad:
            # causal_segmentation_attention_mask.register_hook(lambda grad: torch.clamp(torch.nan_to_num(grad, nan=0), min=-1e3, max=1e3))
            causal_segmentation_attention_mask.register_hook(lambda grad: torch.nan_to_num(grad, nan=0))
        if abstract_causal_segmentation_attention_mask.requires_grad:
            abstract_causal_segmentation_attention_mask.register_hook(lambda grad: torch.clamp(torch.nan_to_num(grad, nan=0), min=-1e3, max=1e3))

        normalized_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)
        return normalized_weights, None, torch.log(causal_segmentation_attention_mask), torch.log(abstract_causal_segmentation_attention_mask)

    def set_temperature(self, temperature):
        self.temperature = temperature

    def set_time_loss_weight(self, time_loss_weight):
        self.time_loss_weight = time_loss_weight

    def set_state_kl_weight(self, state_kl_weight):
        self.state_kl_weight = state_kl_weight

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

class VideoSegmentationModel(FullPrototypeModel):
    def __init__(self, cfg, img_shape, max_seq_len):
        super().__init__(cfg, 1, max_seq_len)
        self.img_shape = img_shape
        self.encoder = CNNEncoder(img_shape)
        self.encoding_dim = self.encoder.get_output_dim()
        self.decoder = CNNDecoder(self.state_rep_dim, img_shape)

        self.context_gru = nn.GRU(self.encoding_dim, self.cfg.context_gru_hidden, batch_first=True)
        self.gru_init = StandardMLP(input_dim=self.cfg.context_gru_hidden, layer_sizes=[256, 256], output_dim=self.cfg.segmentation_transformer_dim)
        self.abstract_init = StandardMLP(input_dim=self.cfg.context_gru_hidden, layer_sizes=[256, 256], output_dim=self.cfg.abstract_rep_deter_dim)

    def forward(self, context, frames, abstract_sample_std_scalar=1.0, state_sample_std_scalar=1.0):
        # traj is a tensor of shape (batch_size, seq_len, data_dim)
        traj = frames
        batch_size = traj.shape[0]
        seq_len = traj.shape[1]
        context_len = context.shape[1]
        device = traj.device
        assert seq_len <= self.max_seq_len, "Trajectories must be less than length {}".format(self.seq_len)

        encodings = self.encoder(traj.reshape(batch_size * seq_len, *self.img_shape))
        encodings = encodings.reshape(batch_size, seq_len, self.encoding_dim)

        context_encodings, _ = self.context_gru(self.encoder(context.reshape(batch_size * context_len, *self.img_shape)).reshape(batch_size, context_len, self.encoding_dim))
        context_encodings = context_encodings[:, -1, :]

        segmentation_encodings = self.segmentation_mlp_encoder(encodings)
        transformed_segmentation_encodings = self.segmentation_transformer_encoder(torch.transpose(segmentation_encodings, 0, 1)).transpose(0, 1) # TRANSPOSE FOR RPR TRANSFORMER
        segmentation_post_logits = []
        segmentation_post_probs = [torch.ones(batch_size, 1, device=device, dtype=torch.float32)]
        segmentation_samples = [torch.ones(batch_size, 1, device=device, dtype=torch.float32)]
        y_samples = []
        gru_hidden = self.gru_init(context_encodings)
        for i in range(1, seq_len):
            # gru_hidden = self.segmentation_gru(torch.cat([segmentation_post_probs[-1], transformed_segmentation_encodings[:, i, :]], dim=-1), gru_hidden)
            # Autoregress with segmentation samples not probs
            gru_hidden = self.segmentation_gru(torch.cat([segmentation_samples[-1], transformed_segmentation_encodings[:, i, :]], dim=-1), gru_hidden)
            segmentation_post_logit = self.segmentation_post(gru_hidden)
            if segmentation_post_logit.requires_grad:
                segmentation_post_logit.retain_grad()
            segmentation_sample, y_sample = concrete.sample_binary_concrete(segmentation_post_logit, self.temperature, hard=self.sample)
            segmentation_post_probs.append(torch.sigmoid(segmentation_post_logit))
            segmentation_post_logits.append(segmentation_post_logit)
            if self.cfg['fix_segmentation_period'] is None:
                segmentation_samples.append(segmentation_sample)
            else:
                if i % self.cfg['fix_segmentation_period'] == 0:
                    segmentation_samples.append(torch.ones_like(segmentation_sample))
                else:
                    segmentation_samples.append(torch.zeros_like(segmentation_sample))
            y_samples.append(y_sample)

        segmentation_post_logit_list = segmentation_post_logits
        segmentation_post_logits = torch.stack(segmentation_post_logits, dim=1)
        segmentation_samples = torch.stack(segmentation_samples, dim=1)
        y_samples = torch.stack(y_samples, dim=1)

        segmentation_samples = segmentation_samples.squeeze(-1)
        if segmentation_samples.requires_grad:
            segmentation_samples.register_hook(lambda grad: self.cfg.time_grad_scalar * grad)
        if segmentation_samples.requires_grad:
            segmentation_samples.retain_grad()
        segment_weights, _, causal_segmentation_attention_mask, abstract_causal_segmentation_attention_mask = self.get_segmentation_attention_masks_probabilistic(segmentation_samples)

        attention_encodings = self.compression_mlp_encoder(torch.cat([encodings, transformed_segmentation_encodings], dim=-1))

        # pre_attention_encodings = self.compression_mlp_encoder(torch.cat([encodings, segmentation_samples.unsqueeze(-1)], dim=-1))
        # transformed_pre_attention_encodings = self.compression_transfomer(torch.transpose(pre_attention_encodings, 0, 1)).transpose(0, 1) # TRANSPOSE FOR RPR TRANSFORMER
        # attention_encodings = self.compression_mlp_decoder(transformed_pre_attention_encodings)

        repeated_attention_encodings = torch.cat([attention_encodings] * seq_len, dim=1).reshape(batch_size, seq_len, seq_len, self.cfg.temporal_attention_dim)
        attended_encodings = torch.sum(segment_weights.unsqueeze(-1) * repeated_attention_encodings, dim=2)
        abstract_rep_post_params = self.abstract_rep_post(attended_encodings)
        abstract_rep_post_means, abstract_rep_post_stds = abstract_rep_post_params[..., :self.cfg.abstract_rep_stoch_dim], nn.functional.softplus(abstract_rep_post_params[..., self.cfg.abstract_rep_stoch_dim:])
        abstract_rep_stoch_samples = self.reparameterize_segments(abstract_rep_post_means, abstract_rep_post_stds, segmentation_samples, std_scalar=abstract_sample_std_scalar)

        abstract_init = self.abstract_init(context_encodings.unsqueeze(1))
        abstract_rep_encodings = self.abstract_rep_mlp_encoder(abstract_rep_stoch_samples)
        transformed_abstract_rep_encodings = self.abstract_rep_transformer_encoder(abstract_rep_encodings, mask=abstract_causal_segmentation_attention_mask)
        abstract_rep_deter = self.abstract_rep_mlp_decoder(transformed_abstract_rep_encodings)
        abstract_rep_prior_params = self.abstract_rep_prior(shift_forward(abstract_rep_deter, 1, fill=abstract_init))
        abstract_rep_prior_means, abstract_rep_prior_stds = abstract_rep_prior_params[..., :self.cfg.abstract_rep_stoch_dim], nn.functional.softplus(abstract_rep_prior_params[..., self.cfg.abstract_rep_stoch_dim:])
        abstract_rep = torch.cat([abstract_rep_stoch_samples, abstract_rep_deter], dim=-1)

        state_rep_post_params = self.state_rep_post(torch.cat([encodings, abstract_rep], dim=-1))
        state_rep_post_means, state_rep_post_stds = state_rep_post_params[..., :self.cfg.state_rep_stoch_dim], nn.functional.softplus(state_rep_post_params[..., self.cfg.state_rep_stoch_dim:])
        state_rep_stoch_samples = self.reparameterize(state_rep_post_means, state_rep_post_stds, std_scalar=state_sample_std_scalar)

        state_rep_encodings = self.state_rep_mlp_encoder(torch.cat([state_rep_stoch_samples, abstract_rep], dim=-1))
        transformed_state_rep_encodings = self.state_rep_transformer_encoder(torch.transpose(state_rep_encodings, 0, 1), mask=causal_segmentation_attention_mask).transpose(0, 1) # TRANSPOSE FOR RPR TRANSFORMER
        state_rep_deter = self.state_rep_mlp_decoder(transformed_state_rep_encodings)
        state_rep_prior_params = self.state_rep_prior(torch.cat([shift_forward(state_rep_deter, 1) * (1 - segmentation_samples).unsqueeze(-1), abstract_rep], dim=-1))
        state_rep_prior_means, state_rep_prior_stds = state_rep_prior_params[..., :self.cfg.state_rep_stoch_dim], nn.functional.softplus(state_rep_prior_params[..., self.cfg.state_rep_stoch_dim:])
        state_rep = torch.cat([state_rep_stoch_samples, state_rep_deter], dim=-1)

        decoder_input = torch.cat([state_rep], dim=-1)
        segmentation_prior_logits = self.segmentation_prior(decoder_input)[:, :-1]
        reconstructed_traj = self.decoder(decoder_input.reshape(batch_size * seq_len, -1)).reshape(batch_size, seq_len, *self.img_shape)

        return reconstructed_traj, dict(
            segment_weights=segment_weights,
            segmentation_post_logits=segmentation_post_logits,
            segmentation_post_logit_list=segmentation_post_logit_list,
            segmentation_samples=segmentation_samples,
            y_samples=y_samples,
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
            segmentation_prior_logits=segmentation_prior_logits,
        )

    def generate(self, context, abstract_sample_std_scalar=1, state_sample_std_scalar=1, generation_length=None, given_segmentations=None, given_abstract_stoch=None, given_state_stoch=None, initial_stoch=None, given_abstract_eps=None, given_state_eps=None):
        device = context.device
        batch_size = context.shape[0]
        if generation_length is None:
            generation_length = self.max_seq_len

        segmentation_probs = torch.zeros(batch_size, generation_length, device=device, dtype=torch.float32)
        segmentations = torch.zeros(batch_size, generation_length, device=device, dtype=torch.float32)
        segmentation_probs[:, 0] = 1
        segmentations[:, 0] = 1
        if given_segmentations is not None:
            segmentations = given_segmentations

        abstract_rep = torch.zeros(batch_size, generation_length, self.abstract_rep_dim, device=device, dtype=torch.float32)
        # abstract_rep = torch.zeros(batch_size, generation_length, self.cfg.abstract_rep_stoch_dim, device=device, dtype=torch.float32)
        if given_abstract_stoch is not None:
            abstract_rep[..., :self.cfg.abstract_rep_stoch_dim] = given_abstract_stoch

        context_len = context.shape[1]
        context_encodings, _ = self.context_gru(self.encoder(context.reshape(batch_size * context_len, *self.img_shape)).reshape(batch_size, context_len, self.encoding_dim))
        context_encodings = context_encodings[:, -1, :]
        abstract_init = self.abstract_init(context_encodings.unsqueeze(1))

        state_rep = torch.zeros(batch_size, generation_length, self.state_rep_dim, device=device, dtype=torch.float32)
        if given_state_stoch is not None:
            state_rep[..., :self.cfg.state_rep_stoch_dim] = given_state_stoch

        generated_traj = torch.zeros(batch_size, generation_length, self.data_dim, device=device, dtype=torch.float32)

        abstract_stoch_means = torch.zeros(batch_size, generation_length, self.cfg.abstract_rep_stoch_dim, device=device, dtype=torch.float32)
        state_stoch_means = torch.zeros(batch_size, generation_length, self.cfg.state_rep_stoch_dim, device=device, dtype=torch.float32)

        if given_abstract_eps is None:
            abstract_eps = torch.randn(batch_size, generation_length, self.cfg.abstract_rep_stoch_dim, device=device, dtype=torch.float32)
        else:
            abstract_eps = given_abstract_eps
        if given_state_eps is None:
            state_eps = torch.randn(batch_size, generation_length, self.cfg.state_rep_stoch_dim, device=device, dtype=torch.float32)
        else:
            state_eps = given_state_eps
        abstract_seg_eps = torch.zeros_like(abstract_eps)

        for i in range(generation_length):
            abstract_rep_prior_params = self.abstract_rep_prior(shift_forward(abstract_rep[:, :i + 1, -self.cfg.abstract_rep_deter_dim:], 1, fill=abstract_init)[:, -1:])
            # abstract_rep_prior_params = self.abstract_rep_prior(shift_forward(abstract_rep[:, :i + 1], 1, fill=abstract_init)[:, -1:])
            abstract_rep_prior_means, abstract_rep_prior_stds = abstract_rep_prior_params[..., :self.cfg.abstract_rep_stoch_dim], nn.functional.softplus(abstract_rep_prior_params[..., self.cfg.abstract_rep_stoch_dim:])
            segment = segmentations[:, i:i + 1].unsqueeze(-1)
            if given_abstract_stoch is None:
                if i == 0:
                    abstract_seg_eps[:, i:i + 1] = abstract_eps[:, i:i + 1]
                    if initial_stoch is None:
                        abstract_rep_stoch_samples = abstract_rep_prior_means + abstract_rep_prior_stds * abstract_seg_eps[:, i:i + 1] * abstract_sample_std_scalar
                    else:
                        abstract_rep_stoch_samples = initial_stoch
                else:
                    abstract_seg_eps[:, i:i + 1] = (1 - segment) * abstract_seg_eps[:, i - 1:i] + segment * abstract_eps[:, i:i + 1]
                    abstract_rep_stoch_samples = (1 - segment) * abstract_rep[:, i - 1:i, :self.cfg.abstract_rep_stoch_dim] + segment * (abstract_rep_prior_means + abstract_rep_prior_stds * abstract_seg_eps[:, i:i + 1] * abstract_sample_std_scalar)
                abstract_rep[:, i:i + 1, :self.cfg.abstract_rep_stoch_dim] = abstract_rep_stoch_samples
            abstract_stoch_means[:, i:i + 1] = abstract_rep_prior_means

            segmentation_samples = segmentations[:, :i + 1]
            _, _, causal_segmentation_attention_mask, abstract_causal_segmentation_attention_mask = self.get_segmentation_attention_masks_probabilistic(segmentation_samples)

            abstract_stoch_hist = abstract_rep[:, :i + 1, :self.cfg.abstract_rep_stoch_dim]
            abstract_rep_encodings = self.abstract_rep_mlp_encoder(abstract_stoch_hist)
            transformed_abstract_rep_encodings = self.abstract_rep_transformer_encoder(abstract_rep_encodings, mask=abstract_causal_segmentation_attention_mask)
            abstract_rep_deter = self.abstract_rep_mlp_decoder(transformed_abstract_rep_encodings)
            abstract_rep[:, i:i + 1, -self.cfg.abstract_rep_deter_dim:] = abstract_rep_deter[:, -1:]

            state_rep_prior_params = self.state_rep_prior(torch.cat([(shift_forward(state_rep[:, :i + 1, -self.cfg.state_rep_deter_dim:], 1))[:, -1:] * (1 - segmentation_samples[:, -1:]).unsqueeze(-1), abstract_rep[:, i:i + 1]], dim=-1))
            state_rep_prior_means, state_rep_prior_stds = state_rep_prior_params[..., :self.cfg.state_rep_stoch_dim], nn.functional.softplus(state_rep_prior_params[..., self.cfg.state_rep_stoch_dim:])
            if given_state_stoch is None:
                state_rep_stoch = state_rep_prior_means + state_rep_prior_stds * state_eps[:, i:i + 1] * state_sample_std_scalar
                state_rep[:, i:i + 1, :self.cfg.state_rep_stoch_dim] = state_rep_stoch
            state_stoch_means[:, i:i + 1] = state_rep_prior_means

            state_stoch_hist = state_rep[:, :i + 1, :self.cfg.state_rep_stoch_dim]
            abstract_rep_hist = abstract_rep[:, :i + 1]
            state_rep_encodings = self.state_rep_mlp_encoder(torch.cat([state_stoch_hist, abstract_rep_hist], dim=-1))
            transformed_state_rep_encodings = self.state_rep_transformer_encoder(torch.transpose(state_rep_encodings, 0, 1), mask=causal_segmentation_attention_mask).transpose(0, 1)
            state_rep_deter = self.state_rep_mlp_decoder(transformed_state_rep_encodings)
            state_rep[:, i:i + 1, -self.cfg.state_rep_deter_dim:] = state_rep_deter[:, -1:]

            decoder_input = torch.cat([state_rep[:, i:i + 1]], dim=-1)
            # generated_traj[:, i:i + 1] = self.decoder(decoder_input)

            if i < generation_length - 1:
                segmentation_prior_logits = self.segmentation_prior(decoder_input)
                segmentation_samples = torch.distributions.Bernoulli(logits=segmentation_prior_logits).sample().squeeze(-1)
                segmentation_probs[:, i + 1:i + 2] = torch.sigmoid(segmentation_prior_logits).squeeze(-1)
                if given_segmentations is None:
                    if self.cfg['fix_segmentation_period'] is None:
                        segmentations[:, i + 1:i + 2] = segmentation_samples
                    else:
                        if (i + 1) % self.cfg['fix_segmentation_period'] == 0:
                            segmentations[:, i + 1:i + 2] = torch.ones_like(segmentation_samples)
                        else:
                            segmentations[:, i + 1:i + 2] = torch.zeros_like(segmentation_samples)

        generated_traj = self.decoder(state_rep.reshape(batch_size * generation_length, -1)).reshape(batch_size, generation_length, *self.img_shape)

        return generated_traj, dict(
            abstract_rep=abstract_rep,
            state_rep=state_rep,
            segmentation_samples=segmentations,
            segmentation_probs=segmentation_probs,
            abstract_stoch_means=abstract_stoch_means,
            state_stoch_means=state_stoch_means,
        )

    def get_loss(self, context, frames):
        reconstructed_frames, info = self.forward(context, frames)
        info['reconstructed_frames'] = reconstructed_frames
        info['ground_truth_frames'] = frames
        reconstruction_loss = nn.functional.mse_loss(frames, reconstructed_frames)
        segmentation_samples = info['segmentation_samples']
        average_compression = 1 / torch.mean(segmentation_samples[:, 1:])

        abstract_rep_post_means, abstract_rep_post_stds = info['abstract_rep_post_means'], info['abstract_rep_post_stds']
        abstract_rep_prior_means, abstract_rep_prior_stds = info['abstract_rep_prior_means'], info['abstract_rep_prior_stds']
        # abstract_rep_prior_means, abstract_rep_prior_stds = torch.zeros_like(abstract_rep_post_means), torch.ones_like(abstract_rep_post_stds)

        # TODO: Don't include time loss factor into KL loss, keep them factorized
        # abstract_rep_kl_loss = torch.mean((self.kl_balance_gaussian(abstract_rep_prior_means, abstract_rep_prior_stds, abstract_rep_post_means, abstract_rep_post_stds, self.cfg.abstract_kl_balance)) * segmentation_samples)
        abs_kl = self.kl_balance_gaussian(abstract_rep_prior_means, abstract_rep_prior_stds, abstract_rep_post_means, abstract_rep_post_stds, self.cfg.abstract_kl_balance)
        abstract_rep_kl_loss = torch.mean(torch.sum(abs_kl * segmentation_samples.detach(), dim=1) / torch.sum(segmentation_samples, dim=1))

        state_rep_post_means, state_rep_post_stds = info['state_rep_post_means'], info['state_rep_post_stds']
        state_rep_prior_means, state_rep_prior_stds = info['state_rep_prior_means'], info['state_rep_prior_stds']

        state_kl = self.kl_balance_gaussian(state_rep_prior_means, state_rep_prior_stds, state_rep_post_means, state_rep_post_stds, self.cfg.state_kl_balance)
        state_rep_kl_loss = torch.mean(state_kl)
        info['state_kl'] = state_kl

        segmentation_post_logits = info['segmentation_post_logits']
        segmentation_prior_logits = info['segmentation_prior_logits']
        segmentation_loss = torch.sigmoid(segmentation_post_logits).mean()
        temp = torch.tensor(self.temperature, device=segmentation_post_logits.device)
        segmentation_kl_loss = torch.mean(concrete.y_kl_divergence(info['y_samples'], segmentation_prior_logits, temp, segmentation_post_logits, temp, kl_balance=self.cfg.segmentation_kl_balance))

        model_loss = self.cfg.reconstruction_loss_weight * reconstruction_loss + \
            self.time_loss_weight * segmentation_loss + \
            self.cfg.abstract_transition_kl_weight * abstract_rep_kl_loss + \
            self.state_kl_weight * state_rep_kl_loss + \
            self.cfg.segmentation_kl_weight * segmentation_kl_loss

        metrics = dict(
            loss=model_loss.item(),
            reconstruction_loss=reconstruction_loss.item(),
            segmentation_loss=segmentation_loss.item(),
            average_compression=torch.minimum(average_compression, torch.tensor(self.max_seq_len, device=average_compression.device)).item(),
            abstract_transition_kl_loss=abstract_rep_kl_loss.item(),
            state_transition_kl_loss=state_rep_kl_loss.item(),
            segmentation_kl_loss=segmentation_kl_loss.item(),
        )

        return model_loss, metrics, info

class FrozenPosteriorVideoSegmentationModel(VideoSegmentationModel):
    def __init__(self, cfg, img_shape, max_seq_len):
        super().__init__(cfg, img_shape, max_seq_len)
        loaded_model = torch.load(cfg.frozen_model_path, map_location='cpu')
        self.segmentation_encoder = loaded_model.encoder
        self.segmentation_context_gru = loaded_model.context_gru
        self.segmentation_mlp_encoder = loaded_model.segmentation_mlp_encoder
        self.segmentation_transformer_encoder = loaded_model.segmentation_transformer_encoder
        self.segmentation_gru = loaded_model.segmentation_gru
        self.segmentation_post = loaded_model.segmentation_post
        self.gru_init = loaded_model.gru_init

    def forward(self, context, frames, abstract_sample_std_scalar=1.0, state_sample_std_scalar=1.0):
        # traj is a tensor of shape (batch_size, seq_len, data_dim)
        traj = frames
        batch_size = traj.shape[0]
        seq_len = traj.shape[1]
        context_len = context.shape[1]
        device = traj.device
        assert seq_len <= self.max_seq_len, "Trajectories must be less than length {}".format(self.seq_len)

        encodings = self.encoder(traj.reshape(batch_size * seq_len, *self.img_shape))
        encodings = encodings.reshape(batch_size, seq_len, self.encoding_dim)

        context_encodings, _ = self.context_gru(self.encoder(context.reshape(batch_size * context_len, *self.img_shape)).reshape(batch_size, context_len, self.encoding_dim))
        context_encodings = context_encodings[:, -1, :]

        segmentation_embeddings = self.segmentation_encoder(traj.reshape(batch_size * seq_len, *self.img_shape)).reshape(batch_size, seq_len, self.encoding_dim)
        segmentation_encodings = self.segmentation_mlp_encoder(segmentation_embeddings)
        transformed_segmentation_encodings = self.segmentation_transformer_encoder(torch.transpose(segmentation_encodings, 0, 1)).transpose(0, 1) # TRANSPOSE FOR RPR TRANSFORMER
        segmentation_post_logits = []
        segmentation_post_probs = [torch.ones(batch_size, 1, device=device, dtype=torch.float32)]
        segmentation_samples = [torch.ones(batch_size, 1, device=device, dtype=torch.float32)]
        y_samples = []
        segmentation_context_encodings, _ = self.segmentation_context_gru(self.segmentation_encoder(context.reshape(batch_size * context_len, *self.img_shape)).reshape(batch_size, context_len, self.encoding_dim))
        segmentation_context_encodings = segmentation_context_encodings[:, -1, :]
        gru_hidden = self.gru_init(segmentation_context_encodings)
        for i in range(1, seq_len):
            # gru_hidden = self.segmentation_gru(torch.cat([segmentation_post_probs[-1], transformed_segmentation_encodings[:, i, :]], dim=-1), gru_hidden)
            # Autoregress with segmentation samples not probs
            gru_hidden = self.segmentation_gru(torch.cat([segmentation_samples[-1], transformed_segmentation_encodings[:, i, :]], dim=-1), gru_hidden)
            segmentation_post_logit = self.segmentation_post(gru_hidden)
            if segmentation_post_logit.requires_grad:
                segmentation_post_logit.retain_grad()
            segmentation_sample, y_sample = concrete.sample_binary_concrete(segmentation_post_logit, self.temperature, hard=True) # FROZEN POSTERIOR, USE HARD SAMPLES
            segmentation_post_probs.append(torch.sigmoid(segmentation_post_logit))
            segmentation_post_logits.append(segmentation_post_logit)
            if self.cfg['fix_segmentation_period'] is None:
                segmentation_samples.append(segmentation_sample)
            else:
                if i % self.cfg['fix_segmentation_period'] == 0:
                    segmentation_samples.append(torch.ones_like(segmentation_sample))
                else:
                    segmentation_samples.append(torch.zeros_like(segmentation_sample))
            y_samples.append(y_sample)

        segmentation_post_logit_list = segmentation_post_logits
        segmentation_post_logits = torch.stack(segmentation_post_logits, dim=1)
        segmentation_samples = torch.stack(segmentation_samples, dim=1)
        y_samples = torch.stack(y_samples, dim=1)

        segmentation_samples = segmentation_samples.squeeze(-1)
        segmentation_samples = segmentation_samples.detach()
        segmentation_post_logits = segmentation_post_logits.detach()
        y_samples = y_samples.detach()
        segment_weights, _, causal_segmentation_attention_mask, _ = self.get_segmentation_attention_masks_probabilistic(segmentation_samples)

        # query_encodings = self.query_mlp_encoder(torch.cat([encodings, broadcast_positional_encoding], dim=-1))
        query_encodings = self.query_mlp_encoder(torch.cat([encodings, transformed_segmentation_encodings.detach()], dim=-1))
        repeated_query_encodings = torch.cat([query_encodings] * seq_len, dim=1).reshape(batch_size, seq_len, seq_len, self.cfg.query_attention_dim)
        attended_query_encodings = torch.sum(segment_weights.unsqueeze(-1) * repeated_query_encodings, dim=2)
        abstract_rep_post_params = self.abstract_rep_post(attended_query_encodings)
        abstract_rep_post_means, abstract_rep_post_stds = abstract_rep_post_params[..., :self.cfg.abstract_rep_stoch_dim], nn.functional.softplus(abstract_rep_post_params[..., self.cfg.abstract_rep_stoch_dim:])
        abstract_rep_stoch_samples = self.reparameterize_segments(abstract_rep_post_means, abstract_rep_post_stds, segmentation_samples, std_scalar=abstract_sample_std_scalar)

        abstract_init = self.abstract_init(context_encodings.unsqueeze(1))
        abstract_rep_prior_params = self.abstract_rep_prior(shift_forward(abstract_rep_stoch_samples, 1, fill=abstract_init))
        abstract_rep_prior_means, abstract_rep_prior_stds = abstract_rep_prior_params[..., :self.cfg.abstract_rep_stoch_dim], nn.functional.softplus(abstract_rep_prior_params[..., self.cfg.abstract_rep_stoch_dim:])
        abstract_rep = abstract_rep_stoch_samples

        state_rep_post_params = self.state_rep_post(torch.cat([encodings, abstract_rep], dim=-1))
        state_rep_post_means, state_rep_post_stds = state_rep_post_params[..., :self.cfg.state_rep_stoch_dim], nn.functional.softplus(state_rep_post_params[..., self.cfg.state_rep_stoch_dim:])
        state_rep_stoch_samples = self.reparameterize(state_rep_post_means, state_rep_post_stds, std_scalar=state_sample_std_scalar)

        state_rep_encodings = self.state_rep_mlp_encoder(torch.cat([state_rep_stoch_samples, abstract_rep], dim=-1))
        transformed_state_rep_encodings = self.state_rep_transformer_encoder(torch.transpose(state_rep_encodings, 0, 1), mask=causal_segmentation_attention_mask).transpose(0, 1) # TRANSPOSE FOR RPR TRANSFORMER
        state_rep_deter = self.state_rep_mlp_decoder(transformed_state_rep_encodings)
        state_rep_prior_params = self.state_rep_prior(torch.cat([shift_forward(state_rep_deter, 1) * (1 - segmentation_samples).unsqueeze(-1), abstract_rep], dim=-1))
        state_rep_prior_means, state_rep_prior_stds = state_rep_prior_params[..., :self.cfg.state_rep_stoch_dim], nn.functional.softplus(state_rep_prior_params[..., self.cfg.state_rep_stoch_dim:])
        state_rep = torch.cat([state_rep_stoch_samples, state_rep_deter], dim=-1)

        decoder_input = torch.cat([state_rep, abstract_rep], dim=-1)
        segmentation_prior_logits = self.segmentation_prior(decoder_input)[:, :-1]
        reconstructed_traj = self.decoder(decoder_input.reshape(batch_size * seq_len, -1)).reshape(batch_size, seq_len, *self.img_shape)

        return reconstructed_traj, dict(
            segment_weights=segment_weights,
            segmentation_post_logits=segmentation_post_logits,
            segmentation_post_logit_list=segmentation_post_logit_list,
            segmentation_samples=segmentation_samples,
            y_samples=y_samples,
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
            segmentation_prior_logits=segmentation_prior_logits,
        )

class RLSegmentationModel(FullPrototypeModel):
    def __init__(self, cfg, obs_dim, action_dim, max_seq_len):
        super().__init__(cfg, obs_dim + action_dim, max_seq_len)
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.state_rep_mlp_encoder = StandardMLP(input_dim=cfg.state_rep_stoch_dim + self.abstract_rep_dim + obs_dim, **cfg.state_rep_mlp_encoder_params, output_dim=cfg.state_rep_transformer_dim)
        self.decoder = StandardMLP(input_dim=self.state_rep_dim + self.abstract_rep_dim + obs_dim, **cfg.decoder_params, output_dim=action_dim)
        self.segmentation_prior = StandardMLP(input_dim=self.state_rep_dim + self.abstract_rep_dim + obs_dim, **cfg.segmentation_prior_params, output_dim=1)

        self.gru_init = StandardMLP(input_dim=self.cfg.segmentation_transformer_dim, layer_sizes=[256, 256], output_dim=self.cfg.segmentation_transformer_dim)
        self.abstract_init = StandardMLP(input_dim=obs_dim, layer_sizes=[256, 256], output_dim=self.cfg.abstract_rep_stoch_dim)

    def forward(self, obs, act, abstract_sample_std_scalar=1.0, state_sample_std_scalar=1.0):
        # traj is a tensor of shape (batch_size, seq_len, data_dim)
        traj = torch.cat([obs, act], dim=-1)
        batch_size = traj.shape[0]
        seq_len = traj.shape[1]
        device = traj.device
        assert seq_len <= self.max_seq_len, "Trajectories must be less than length {}".format(self.seq_len)
        encodings = self.encoder(traj)
        broadcast_positional_encoding = self.positional_encoding_dropout(self.positional_encoding[:, :traj.shape[1]].expand(batch_size, -1, -1))

        segmentation_encodings = self.segmentation_mlp_encoder(encodings)
        transformed_segmentation_encodings = self.segmentation_transformer_encoder(torch.transpose(segmentation_encodings, 0, 1)).transpose(0, 1) # TRANSPOSE FOR RPR TRANSFORMER
        segmentation_post_logits = []
        segmentation_post_probs = [torch.ones(batch_size, 1, device=device, dtype=torch.float32)]
        segmentation_samples = [torch.ones(batch_size, 1, device=device, dtype=torch.float32)]
        y_samples = []
        # gru_hidden = torch.zeros(batch_size, self.cfg.segmentation_transformer_dim, device=device, dtype=torch.float32)
        gru_hidden = self.gru_init(transformed_segmentation_encodings[:, 0, :])
        for i in range(1, seq_len):
            # gru_hidden = self.segmentation_gru(torch.cat([segmentation_post_probs[-1], transformed_segmentation_encodings[:, i, :]], dim=-1), gru_hidden)
            # Autoregress with segmentation samples not probs
            gru_hidden = self.segmentation_gru(torch.cat([segmentation_samples[-1], transformed_segmentation_encodings[:, i, :]], dim=-1), gru_hidden)
            segmentation_post_logit = self.segmentation_post(gru_hidden)
            if segmentation_post_logit.requires_grad:
                segmentation_post_logit.retain_grad()
            segmentation_sample, y_sample = concrete.sample_binary_concrete(segmentation_post_logit, self.temperature, hard=self.sample)
            segmentation_post_probs.append(torch.sigmoid(segmentation_post_logit))
            segmentation_post_logits.append(segmentation_post_logit)
            if self.cfg['fix_segmentation_period'] is None:
                segmentation_samples.append(segmentation_sample)
            else:
                if i % self.cfg['fix_segmentation_period'] == 0:
                    segmentation_samples.append(torch.ones_like(segmentation_sample))
                else:
                    segmentation_samples.append(torch.zeros_like(segmentation_sample))
            y_samples.append(y_sample)

        segmentation_post_logit_list = segmentation_post_logits
        segmentation_post_logits = torch.stack(segmentation_post_logits, dim=1)
        segmentation_samples = torch.stack(segmentation_samples, dim=1)
        y_samples = torch.stack(y_samples, dim=1)

        segmentation_samples = segmentation_samples.squeeze(-1)
        if segmentation_samples.requires_grad:
            segmentation_samples.register_hook(lambda grad: self.cfg.time_grad_scalar * grad)
        if segmentation_samples.requires_grad:
            segmentation_samples.retain_grad()
        segment_weights, _, causal_segmentation_attention_mask, _ = self.get_segmentation_attention_masks_probabilistic(segmentation_samples)

        # query_encodings = self.query_mlp_encoder(torch.cat([encodings, broadcast_positional_encoding], dim=-1))
        query_encodings = self.query_mlp_encoder(torch.cat([encodings, transformed_segmentation_encodings], dim=-1))
        repeated_query_encodings = torch.cat([query_encodings] * seq_len, dim=1).reshape(batch_size, seq_len, seq_len, self.cfg.query_attention_dim)
        attended_query_encodings = torch.sum(segment_weights.unsqueeze(-1) * repeated_query_encodings, dim=2)
        abstract_rep_post_params = self.abstract_rep_post(attended_query_encodings)
        abstract_rep_post_means, abstract_rep_post_stds = abstract_rep_post_params[..., :self.cfg.abstract_rep_stoch_dim], nn.functional.softplus(abstract_rep_post_params[..., self.cfg.abstract_rep_stoch_dim:])
        abstract_rep_stoch_samples = self.reparameterize_segments(abstract_rep_post_means, abstract_rep_post_stds, segmentation_samples, std_scalar=abstract_sample_std_scalar)

        abstract_init = self.abstract_init(obs[:, 0:1, :])
        abstract_rep_prior_params = self.abstract_rep_prior(shift_forward(abstract_rep_stoch_samples, 1, fill=abstract_init))
        abstract_rep_prior_means, abstract_rep_prior_stds = abstract_rep_prior_params[..., :self.cfg.abstract_rep_stoch_dim], nn.functional.softplus(abstract_rep_prior_params[..., self.cfg.abstract_rep_stoch_dim:])
        abstract_rep = abstract_rep_stoch_samples

        state_rep_post_params = self.state_rep_post(torch.cat([encodings, abstract_rep], dim=-1))
        state_rep_post_means, state_rep_post_stds = state_rep_post_params[..., :self.cfg.state_rep_stoch_dim], nn.functional.softplus(state_rep_post_params[..., self.cfg.state_rep_stoch_dim:])
        state_rep_stoch_samples = self.reparameterize(state_rep_post_means, state_rep_post_stds, std_scalar=state_sample_std_scalar)

        state_rep_encodings = self.state_rep_mlp_encoder(torch.cat([state_rep_stoch_samples, abstract_rep, obs], dim=-1))
        transformed_state_rep_encodings = self.state_rep_transformer_encoder(torch.transpose(state_rep_encodings, 0, 1), mask=causal_segmentation_attention_mask).transpose(0, 1) # TRANSPOSE FOR RPR TRANSFORMER
        state_rep_deter = self.state_rep_mlp_decoder(transformed_state_rep_encodings)
        state_rep_prior_params = self.state_rep_prior(torch.cat([shift_forward(state_rep_deter, 1) * (1 - segmentation_samples).unsqueeze(-1), abstract_rep], dim=-1))
        state_rep_prior_means, state_rep_prior_stds = state_rep_prior_params[..., :self.cfg.state_rep_stoch_dim], nn.functional.softplus(state_rep_prior_params[..., self.cfg.state_rep_stoch_dim:])
        state_rep = torch.cat([state_rep_stoch_samples, state_rep_deter], dim=-1)

        # state_rep = torch.zeros_like(state_rep)
        decoder_input = torch.cat([state_rep, abstract_rep, obs], dim=-1)
        segmentation_prior_logits = self.segmentation_prior(decoder_input)[:, :-1]
        reconstructed_traj = self.decoder(decoder_input)

        return reconstructed_traj, dict(
            segment_weights=segment_weights,
            segmentation_post_logits=segmentation_post_logits,
            segmentation_post_logit_list=segmentation_post_logit_list,
            segmentation_samples=segmentation_samples,
            y_samples=y_samples,
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
            segmentation_prior_logits=segmentation_prior_logits,
        )

    def get_loss(self, obs, act):
        reconstructed_act, info = self.forward(obs, act)
        info['reconstructed_act'] = reconstructed_act
        info['ground_truth_act'] = act
        info['ground_truth_obs'] = obs
        reconstruction_loss = nn.functional.mse_loss(act, reconstructed_act)
        segmentation_samples = info['segmentation_samples']
        average_compression = 1 / torch.mean(segmentation_samples[:, 1:])

        abstract_rep_post_means, abstract_rep_post_stds = info['abstract_rep_post_means'], info['abstract_rep_post_stds']
        abstract_rep_prior_means, abstract_rep_prior_stds = info['abstract_rep_prior_means'], info['abstract_rep_prior_stds']
        # abstract_rep_prior_means, abstract_rep_prior_stds = torch.zeros_like(abstract_rep_post_means), torch.ones_like(abstract_rep_post_stds)

        # TODO: Don't include time loss factor into KL loss, keep them factorized
        # abstract_rep_kl_loss = torch.mean((self.kl_balance_gaussian(abstract_rep_prior_means, abstract_rep_prior_stds, abstract_rep_post_means, abstract_rep_post_stds, self.cfg.abstract_kl_balance)) * segmentation_samples)
        abs_kl = self.kl_balance_gaussian(abstract_rep_prior_means, abstract_rep_prior_stds, abstract_rep_post_means, abstract_rep_post_stds, self.cfg.abstract_kl_balance)
        abstract_rep_kl_loss = torch.mean(torch.sum(abs_kl * segmentation_samples.detach(), dim=1) / torch.sum(segmentation_samples, dim=1))

        state_rep_post_means, state_rep_post_stds = info['state_rep_post_means'], info['state_rep_post_stds']
        state_rep_prior_means, state_rep_prior_stds = info['state_rep_prior_means'], info['state_rep_prior_stds']

        state_kl = self.kl_balance_gaussian(state_rep_prior_means, state_rep_prior_stds, state_rep_post_means, state_rep_post_stds, self.cfg.state_kl_balance)
        state_rep_kl_loss = torch.mean(state_kl)
        info['state_kl'] = state_kl

        segmentation_post_logits = info['segmentation_post_logits']
        segmentation_prior_logits = info['segmentation_prior_logits']
        segmentation_loss = torch.sigmoid(segmentation_post_logits).mean()
        temp = torch.tensor(self.temperature, device=segmentation_post_logits.device)
        segmentation_kl_loss = torch.mean(concrete.y_kl_divergence(info['y_samples'], segmentation_prior_logits, temp, segmentation_post_logits, temp, kl_balance=self.cfg.segmentation_kl_balance))

        model_loss = self.cfg.reconstruction_loss_weight * reconstruction_loss + \
            self.time_loss_weight * segmentation_loss + \
            self.cfg.abstract_transition_kl_weight * abstract_rep_kl_loss + \
            self.state_kl_weight * state_rep_kl_loss + \
            self.cfg.segmentation_kl_weight * segmentation_kl_loss

        metrics = dict(
            loss=model_loss.item(),
            reconstruction_loss=reconstruction_loss.item(),
            segmentation_loss=segmentation_loss.item(),
            average_compression=torch.minimum(average_compression, torch.tensor(self.max_seq_len, device=average_compression.device)).item(),
            abstract_transition_kl_loss=abstract_rep_kl_loss.item(),
            state_transition_kl_loss=state_rep_kl_loss.item(),
            segmentation_kl_loss=segmentation_kl_loss.item(),
        )

        return model_loss, metrics, info

def get_activation(activation):
    if activation == 'elu':
        return nn.ELU
    elif activation == 'relu':
        return nn.ReLU
    else:
        return NotImplementedError("Activation not implemented yet")

def shift_forward(x, shift, fill=None):
    if fill is None:
        return torch.cat([torch.zeros_like(x[:, -shift:]), x[:, :-shift]], dim=1)
    else:
        assert fill.shape[1] == shift
        return torch.cat([fill, x[:, :-shift]], dim=1)

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

class CNNEncoder(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        self.img_shape = img_shape

        self.feature = nn.Sequential(
            nn.Conv2d(img_shape[0], 32, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.reshape(x.shape[0], -1)
        return x

    def get_output_dim(self):
        return 1024

class CNNDecoder(nn.Module):
    def __init__(self, feat_dim, img_shape):
        super().__init__()
        self.feat_dim = feat_dim
        self.img_shape = img_shape

        self.linear = nn.Linear(feat_dim, 1024)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_shape[0], 6, stride=2)
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 1, 1)
        mean = self.decoder(x)
        return mean

    def get_mse(self, feats, batch_obs):
        mean = self.forward(feats)
        loss = nn.MSELoss()(mean, batch_obs)
        return loss

def get_sinusoidal_positional_encoding(dim, max_len):
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2) * (-np.log(10000.0) / dim))
    pe = torch.zeros(1, max_len, dim)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    return pe

# Large portions of this transformer stuff taken from the pytorch transformer implmentation
class GivenQueryTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_query,
        d_model,
        batch_first=True,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        layer_norm_eps=1e-5,
        norm_first=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.query_map = nn.Linear(d_query, d_model, **factory_kwargs)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)

        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.functional.relu

    def forward(self, query, x, src_mask=None):
        if self.norm_first:
            x = self._sa_block(query, self.norm1(x), src_mask)
            x = self._ff_block(self.norm2(x))
        else:
            x = self.norm1(self._sa_block(x, src_mask))
            x = self.norm2(self._ff_block(x))
        return x

    # self-attention block
    def _sa_block(self, query, x, attn_mask):
        query = self.query_map(query)
        x = self.self_attn(query, x, x,
                           attn_mask=attn_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class GivenQueryTransformerEncoder(nn.Module):
    def __init__(self, given_query_encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(given_query_encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, query, src, mask=None):
        output = src

        for mod in self.layers:
            output = mod(query, output, src_mask=mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

def prepend_null_token_transformer_encoder_pass(transformer_input, transformer_mask, transformer, nheads, null_token=None, null_mask_value=0):
    # Expecting batch first input, so (batch_size, seq_len, input_dim)
    batch_size, seq_len, _ = transformer_input.shape
    device = transformer_input.device
    if null_token is None:
        null_token = torch.zeros_like(transformer_input[:, :1, :])
    null_vector_to_others_mask = -torch.inf * torch.ones(batch_size * nheads, 1, seq_len, device=device)
    all_vectors_to_null_mask = torch.ones(batch_size * nheads, seq_len + 1, 1, device=device) * null_mask_value

    transformer_input = torch.cat([null_token, transformer_input], dim=1)
    transformer_mask = torch.cat([all_vectors_to_null_mask, torch.cat([null_vector_to_others_mask, transformer_mask], dim=1)], dim=2)
    transformer_output = transformer(transformer_input, mask=transformer_mask)
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
    # mask = torch.where(torch.rand(batch_size, sequence_len, sequence_len) > 0.5, -torch.inf * torch.ones(1), torch.zeros(1))
    mask = torch.cat([torch.zeros(batch_size, sequence_len, 1), torch.cat([torch.ones(batch_size, 1, sequence_len - 1) * -torch.inf, torch.ones(batch_size, sequence_len - 1, sequence_len - 1) * -torch.inf], dim=-2)], dim=-1)
    mask = mask.unsqueeze(1).repeat(1, num_heads, 1, 1).reshape(batch_size * num_heads, sequence_len, sequence_len)
    print("Mask: ")
    print(mask)
    print(mask.shape)
    print("Reshaped mask: ")
    print(mask.reshape(batch_size, num_heads, sequence_len, sequence_len))
    print(mask.reshape(batch_size, num_heads, sequence_len, sequence_len).shape)
    output, weights = mha(q, k, v, attn_mask=mask)
    print("Weights: ")
    print(weights)
    print(weights.shape)
    print("Output: ")
    print(output)
    print(output.shape)

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

def test_full_prototype_generation():
    from hydra import initialize, compose

    batch_size = 2
    seq_len = 5
    data_dim = 1

    with initialize(version_base="1.3", config_path="cfgs/model/"):
        cfg = compose(config_name="full_prototype_v1")
        model = FullPrototypeModel(cfg, data_dim=data_dim, max_seq_len=seq_len)

        generation, _ = model.generate(2, generation_length=seq_len)
        print("Generation reconstruction:", generation)

if __name__ == '__main__':
    # test_temporal_attention()
    # multihead_attention_mask_shape_test()
    test_full_prototype_forward()
    test_full_prototype_generation()
    pass
