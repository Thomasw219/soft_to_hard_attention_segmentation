import numpy as np
import argparse
import os
import pdb
import pickle
import torch
import gym
import math

from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F

from itertools import product
from digail.models import Policy, Posterior, DiscretePosterior

#-----Environment-----#

#if args.expert_path == 'SR2_expert_trajectories/':
#    R = RewardFunction_SR2(-1.0,1.0,width)
#else:
#    R = RewardFunction(-1.0,1.0)

global_args = None

class VAE(nn.Module):
    def __init__(self,
                 policy_state_size=1, posterior_state_size=1,
                 policy_action_size=1, posterior_action_size=1,
                 policy_latent_size=1, posterior_latent_size=1,
                 posterior_goal_size=1,
                 policy_output_size=1,
                 history_size=1,
                 hidden_size=64,
        ):
        '''
        state_size: State size
        latent_size: Size of 'c' variable
        goal_size: Number of goals.
        output_size:
        '''
        super(VAE, self).__init__()

        self.history_size = history_size
        self.policy_state_size = policy_state_size
        self.posterior_latent_size = posterior_latent_size
        self.posterior_goal_size = posterior_goal_size

        self.policy_latent_size = policy_latent_size

        #if args.discrete:
        #    output_activation='sigmoid'
        #else:
        output_activation=None

        self.policy = Policy(state_size=policy_state_size,
                             action_size=policy_action_size,
                             latent_size=self.policy_latent_size,
                             output_size=policy_output_size,
                             hidden_size=hidden_size,
                             history_size=history_size,
                             output_activation=output_activation)

        self.posterior = Posterior(
                state_size=posterior_state_size*self.history_size,
                action_size=posterior_action_size,
                latent_size=posterior_latent_size,
                output_size=posterior_latent_size,
                hidden_size=hidden_size)


    def encode(self, x, c):
        return self.posterior(torch.cat((x, c), 1))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu


    def decode_goal_policy(self, x, g):
        action_mean, _, _ = self.policy_goal(torch.cat((x, g), 1))
        if 'circle' in global_args.env_type:
            action_mean = action_mean / torch.norm(action_mean, dim=1).unsqueeze(1)
        return action_mean

    def decode(self, x, c):
        action_mean, action_log_std, action_std = self.policy(
                torch.cat((x, c), 1))

        return action_mean

    def forward(self, x, c, g, only_goal_policy=False):
        if only_goal_policy:
            decoder_output_2 = self.decode_goal_policy(x, g)
            # Return a tuple as the else part below. Caller should expect a
            # tuple always.
            return decoder_output_2,
        else:
            mu, logvar = self.encode(x, c)
            c[:,-self.posterior_latent_size:] = self.reparameterize(mu, logvar)

            decoder_output_1 = None
            decoder_output_2 = None


        if self.use_goal_in_policy:
            if self.use_history_in_policy:
                decoder_output_1 = self.decode(x, c)
            else:
                decoder_output_1 = self.decode(
                        x[:, -self.policy_state_size:], c)
        else:
            if self.use_history_in_policy:
                decoder_output_1 = self.decode(
                        x, c[:,-self.posterior_latent_size:])
            else:
                decoder_output_1 = self.decode(
                        x[:, -self.policy_state_size:],
                        c[:,-self.posterior_latent_size:])

            if self.use_separate_goal_policy:
                decoder_output_2 = self.decode_goal_policy(x, g)


            return decoder_output_1, decoder_output_2, mu, logvar

class DiscreteVAE(VAE):
    def __init__(
            self,
            temperature=5.0,
            cosine_similarity_loss_weight=50.0,
            reconstruction_loss_coeff=1.0,
            kl_loss_coeff=1.0,
            **kwargs
        ):
        '''
        state_size: State size
        latent_size: Size of 'c' variable
        goal_size: Number of goals.
        output_size:
        '''
        super(DiscreteVAE, self).__init__(**kwargs)
        self.posterior = DiscretePosterior(
                state_size=kwargs['posterior_state_size']*self.history_size,
                action_size=kwargs['posterior_action_size'],
                latent_size=kwargs['posterior_latent_size'],
                output_size=kwargs['posterior_latent_size'],
                hidden_size=kwargs['hidden_size'],
        )
        self.encoder_softmax = nn.Softmax(dim=1)
        self.temperature = temperature
        self.init_temperature = temperature
        self.cosine_similarity_loss_weight = cosine_similarity_loss_weight
        print('cosine_similarity_loss_weight', cosine_similarity_loss_weight)
        self.reconstruction_loss_coeff = reconstruction_loss_coeff
        self.kl_loss_coeff = kl_loss_coeff

    def update_temperature(self, epoch):
        '''Update temperature.'''
        r = 5e-4  # will become 1.0 after 3000 epochs
        # r = 33e-4 will become 1.0 after 500 epochs and 0.18 after 1000 epochs.
        # r = 0.023 # Will become 0.1 after 100 epochs if initial temp is 1.0
        # r = 0.011 # Will become 0.1 after 200 epochs if initial temp is 1.0
        self.temperature = max(0.1, self.init_temperature * math.exp(-r*epoch))


    def encode(self, x, c):
        '''Return the log probability output for the encoder.'''
        logits = self.posterior(torch.cat((x, c), 1))
        return logits

    def sample_gumbel(self, shape, eps=1e-20):
        """Sample from Gumbel(0, 1)"""
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        dtype = logits.data.type()
        y = logits + Variable(self.sample_gumbel(logits.size())).type(dtype).to(logits.device)
        y = F.softmax(y / temperature, dim=1)
        # shape = y.size()
        # _, ind = y.max(dim=-1)
        # y_hard = torch.zeros_like(y).view(-1, shape[-1])
        # y_hard.scatter_(1, ind.view(-1, 1), 1)
        # y_hard = y_hard.view(*shape)
        # return (y_hard - y).detach() + y
        return y

    def reparameterize(self, logits, temperature, eps=1e-10):
        if self.training:
            probs = self.gumbel_softmax_sample(logits, temperature)
        else:
            probs = F.softmax(logits / temperature, dim=1)
        return probs

    def forward(self, x, c):
        c_logits = self.encode(x, c)
        c[:, -self.posterior_latent_size:] = self.reparameterize(
                c_logits, self.temperature)

        decoder_output_1 = None
        decoder_output_2 = None

        decoder_output_1 = self.decode(x, c)

        return decoder_output_1, decoder_output_2, c_logits

    def loss_function(self, recon_x1, recon_x2, x, vae_posterior_output):
        loss1 = F.mse_loss(recon_x1, x, reduce='sum')

        # logits is the un-normalized log probability for belonging to a class
        logits = vae_posterior_output[0]
        num_q_classes = self.posterior_latent_size
        # q_prob is (N, C)
        q_prob = F.softmax(logits, dim=1)  # q_prob
        log_q_prob = torch.log(q_prob + 1e-10)  # log q_prob
        prior_prob = Variable(torch.Tensor([1.0 / num_q_classes])).type(
                logits.data.type()).to(q_prob.device)
        batch_size = logits.size(0)
        KLD = torch.sum(q_prob * (log_q_prob - torch.log(prior_prob))) / batch_size
        # print("q_prob: {}".format(q_prob))

        #return MSE + KLD
        return self.reconstruction_loss_coeff*loss1 + self.kl_loss_coeff*KLD, loss1, None, KLD

    def get_loss(self, obs, actions):
        device = obs.device
        batch_size = obs.shape[0]
        episode_len = obs.shape[1]
        history_size = self.history_size

        train_loss, train_policy_loss = 0.0, 0.0
        train_KLD_loss, train_policy2_loss = 0.0, 0.0
        train_cosine_loss_for_context = 0.0
        ep_timesteps = 0
        true_return = 0.0

        ep_state = obs.numpy(force=True)
        ep_action = actions.numpy(force=True)

        action_var = Variable(torch.from_numpy(ep_action).float().to(device))

        c = -1 * np.ones((batch_size, self.posterior_latent_size), dtype=np.float32)

        x_feat = ep_state[:, 0, :]
        x = x_feat

        # Add history to state
        if history_size > 1:
            x_hist = -1 * np.ones((x.shape[0], history_size, x.shape[1]),
                                    dtype=np.float32)
            x_hist[:, history_size - 1, :] = x_feat
            x = x_hist

        c_var_hist = []

        reconstruction_loss = []
        kld_loss = []
        cosine_similarity_loss = []

        reconstructed_actions = []
        cosine_similarity_losses = []

        # Store list of losses to backprop later.
        ep_loss, curr_state_arr = [], ep_state[:, 0, :]
        for t in range(episode_len):
            ep_timesteps += 1
            x_var = Variable(torch.from_numpy(
                x.reshape((batch_size, -1))).type(obs.dtype).to(device))

            # Append 'c' at the end.
            c_var = Variable(torch.from_numpy(c).type(obs.dtype).to(device))

            vae_output = self.forward(
                    x_var, c_var)
            reconstructed_actions.append(vae_output[0])

            expert_action_var = action_var[:, t, :].clone()
            vae_reparam_input = (vae_output[2],
                                    self.temperature)

            loss, policy_loss, policy2_loss, KLD_loss = \
                    self.loss_function(vae_output[0],
                                        vae_output[1],
                                        expert_action_var,
                                        vae_output[2:],
                                        )

            train_policy_loss += policy_loss
            train_KLD_loss += KLD_loss

            if len(c_var_hist) > 0:
                # Add cosine loss for similarity
                last_c = c_var_hist[-1]
                curr_c = vae_output[-1]
                cos_loss_context = (1.0 - F.cosine_similarity(last_c, curr_c))
                cos_loss_mean= cos_loss_context.mean()
                loss += self.cosine_similarity_loss_weight * cos_loss_mean
                # print("Cosine loss: {}".format(cos_loss_mean.item() * self.cosine_similarity_loss_weight))
                train_cosine_loss_for_context += cos_loss_mean
                cosine_similarity_loss.append(cos_loss_mean.item())
                cosine_similarity_losses.append(cos_loss_context)

            reconstruction_loss.append(policy_loss.item())
            kld_loss.append(KLD_loss.item())

            ep_loss.append(loss)
            train_loss += loss

            c_var_hist.append(vae_output[-1])

            if history_size > 1:
                x_hist[:, :(history_size-1), :] = x_hist[:, 1:, :]

            if t < episode_len-1:
                next_state = ep_state[:, t+1, :]
            else:
                break

            if history_size > 1:
                x_hist[:, history_size-1] = next_state
                x = x_hist
            else:
                x[:] = next_state

            # update c
            c = \
                self.reparameterize(
                        *vae_reparam_input).numpy(force=True)

        # Calculate the total loss.
        total_loss = ep_loss[0]
        for t in range(1, len(ep_loss)):
            total_loss = total_loss + ep_loss[t]

        loss = total_loss / ep_timesteps

        metrics = {
            'loss' : loss.item(),
            'reconstruction_loss' : np.mean(reconstruction_loss),
            'kld_loss' : np.mean(kld_loss),
            'cosine_similarity_loss' : np.mean(cosine_similarity_loss),
            'compression_rate' : 1 / np.maximum(1 / ep_timesteps, np.mean(cosine_similarity_loss)),
        }

        info = {
            'ground_truth_obs' : obs,
            'ground_truth_act' : actions,
            'reconstructed_act' : torch.stack(reconstructed_actions, dim=1),
            'cosine_similarity_losses' : torch.stack(cosine_similarity_losses, dim=1),
        }

        return loss, metrics, info
