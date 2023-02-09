import os
from datetime import datetime, timezone, timedelta
from time import time

import hydra
from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import RandomSizePiecewiseSine as Dataset
from models import FullPrototypeModel
from utils import make_scheduler

@hydra.main(version_base='1.3', config_path='cfgs', config_name='full_prototype_experiment')
def eval_full_prototype(cfg):
    np.random.seed(cfg['np_seed'])
    torch.manual_seed(0)
    train_dataset = Dataset(**cfg['train_dataset'])
    train_dataloader = DataLoader(train_dataset, **cfg['dataloader'])

    test_dataset = Dataset(**cfg['test_dataset'])
    test_dataloader = DataLoader(test_dataset, **cfg['dataloader'])

    model = FullPrototypeModel(cfg['model'], data_dim=1, max_seq_len=cfg['train_dataset']['signal_length'])
    model.to(cfg['device'])
    model.hard_sample()

    model.load_state_dict(torch.load(cfg['model_load_path'], map_location=cfg['device']))

    batch = next(iter(test_dataloader))
    reconstruction, rec_info = model.forward(batch.to(cfg['device'], dtype=torch.float32))
    # print("Segmentation samples:")
    # print(rec_info["segmentation_samples"][0])
    # print("Abstract stoch state:")
    # print(rec_info["abstract_rep"][0, :, :model.cfg.abstract_rep_stoch_dim])
    # print(rec_info["abstract_rep_prior_stds"][0, :, :model.cfg.abstract_rep_stoch_dim])

    # generation, gen_info = model.generate(batch_size=1, generation_length=model.max_seq_len, given_segmentations=rec_info["segmentation_samples"][:1], given_abstract_stoch=rec_info["abstract_rep"][:1, :, :model.cfg.abstract_rep_stoch_dim])
    generation, gen_info = model.generate(batch_size=1, generation_length=model.max_seq_len)
    # print("Abstract stoch prior diff:")
    # print(gen_info["abstract_stoch_means"][0, :, :model.cfg.abstract_rep_stoch_dim] - rec_info["abstract_rep_prior_means"][0, :, :model.cfg.abstract_rep_stoch_dim])
    print("Generation abstract stochastic state:")
    print(torch.cat([gen_info["abstract_rep"][0, :, :model.cfg.abstract_rep_stoch_dim], gen_info["segmentation_samples"][0, :].unsqueeze(-1)], dim=-1))

    visualize_reconstruction_generation(reconstruction[0], generation[0], "figures")

def plt_prep(tensor):
    return tensor.detach().cpu().numpy().squeeze()

def visualize(info, logger, global_step, n_samples=3, prefix='train'):
    plot_fig = plt.figure(0)
    delta_t_fig = plt.figure(1)
    delta_t_logit_fig = plt.figure(2)
    latent_features_fig = plt.figure(3)
    for i in range(n_samples):
        # Plot ground truth and reconstruction for n_samples
        plot_ax = plot_fig.add_subplot(n_samples, 1, i+1)
        gt_traj = plt_prep(info['ground_truth_traj'][i])
        recon_traj = plt_prep(info['reconstructed_traj'][i])
        plot_ax.plot(gt_traj, label='ground truth', c='b')
        plot_ax.plot(recon_traj, label='reconstruction', c='g')
        plot_ax.set_ylim(np.min(np.concatenate([-1 * np.ones_like(gt_traj), gt_traj, recon_traj])),
                         np.max(np.concatenate([np.ones_like(gt_traj), gt_traj, recon_traj])))

        # Plot delta_t for sequence
        delta_t_ax = delta_t_fig.add_subplot(n_samples, 1, i+1)
        delta_t_ax.plot(plt_prep(info['segmentation_samples'][i]), label='delta_t', c='r')

        # Plot delta_t logit for sequence
        delta_t_logit_ax = delta_t_logit_fig.add_subplot(n_samples, 1, i+1)
        delta_t_logit_ax.plot(plt_prep(info['segmentation_logits'][i, :, 0]), label='0 logits', c='b')
        delta_t_logit_ax.plot(plt_prep(info['segmentation_logits'][i, :, 1]), label='1 logits', c='g')
        prior_logit_ax = delta_t_logit_ax.twinx()
        prior_logit_ax.plot(plt_prep(torch.sigmoid(info['segmentation_prior_logits'][i, :, 0])), label='prior prob', c='r')
        prior_logit_ax.set_ylim(0, 1)

        # Plot latent features for sequence
        latent_features_ax = latent_features_fig.add_subplot(n_samples, 1, i+1)
        for j in range(info['abstract_rep'].shape[-1]):
            latent_features_ax.plot(plt_prep(info['abstract_rep'][i, :, j]))

        if i == 0:
            plot_ax.legend()
            delta_t_ax.legend()
            delta_t_logit_ax.legend()
            prior_logit_ax.legend()

    logger.add_figure(prefix + '/reconstruction', plot_fig, global_step)
    logger.add_figure(prefix + '/delta_t', delta_t_fig, global_step)
    logger.add_figure(prefix + '/delta_t_logit', delta_t_logit_fig, global_step)
    logger.add_figure(prefix + '/latent_features', latent_features_fig, global_step)

    plot_fig.clf()
    delta_t_fig.clf()
    delta_t_logit_fig.clf()
    latent_features_fig.clf()

def visualize_generations(model, logger, global_step, n_samples=3, prefix='test'):
    generated_trajs, info = model.generate(n_samples, generation_length=model.max_seq_len)
    plot_fig = plt.figure(0)
    segmentation_prob_fig = plt.figure(1)
    for i in range(n_samples):
        # Plot generated trajectories and discrete segmentation points
        plot_ax = plot_fig.add_subplot(n_samples, 1, i+1)
        plot_ax.plot(plt_prep(generated_trajs[i]), label='generated', zorder=10, c='g')
        segmentations = plt_prep(info['segmentation_samples'][i])
        indices = np.arange(segmentations.shape[0])
        plot_ax.vlines(indices[segmentations == 1], np.min(plt_prep(generated_trajs[i])), np.max(plt_prep(generated_trajs[i])), label='segmentations', zorder=0, color='k')

        segmentation_prob_ax = segmentation_prob_fig.add_subplot(n_samples, 1, i+1)
        segmentation_prob_ax.plot(plt_prep(info['segmentation_probs'][i, :]), label='segmentation probability', c='b')

        if i == 0:
            plot_ax.legend()
            segmentation_prob_ax.legend()

    logger.add_figure(prefix + '/generation', plot_fig, global_step)
    logger.add_figure(prefix + '/generation_segmentation_prob', segmentation_prob_fig, global_step)

    plot_fig.clf()
    segmentation_prob_fig.clf()

def visualize_reconstruction_generation(reconstruction, generation, save_dir):
    plt.plot(plt_prep(reconstruction), label='reconstruction')
    plt.plot(plt_prep(generation), label='generation')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'reconstruction_generation.png'))

if __name__ == '__main__':
    eval_full_prototype()