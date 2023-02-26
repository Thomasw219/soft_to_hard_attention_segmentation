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

from data import SimplePiecewiseLinear as Dataset
from love.love_hssm import EnvModel
from utils import make_scheduler

@hydra.main(version_base='1.3', config_path='cfgs', config_name='love_experiment')
def test_full_prototype(cfg):
    np.random.seed(cfg['np_seed'])
    train_dataset = Dataset(**cfg['train_dataset'])
    train_dataloader = DataLoader(train_dataset, **cfg['dataloader'])

    test_dataset = Dataset(**cfg['test_dataset'])
    test_dataloader = DataLoader(test_dataset, **cfg['dataloader'])

    model = EnvModel(**cfg['model'])
    model.to(cfg['device'])
    optimizer = get_optimizer(cfg['optimizer'], model)
    timestring = datetime.now(tz=timezone(timedelta(hours=-5))).strftime("_%m-%d-%Y_%H-%M-%S") # EST, No daylight savings
    logger = SummaryWriter(os.path.join(cfg['log_dir'], cfg['name'] + timestring))
    logger.add_text('config', str(cfg))
    logger.add_text('model', str(model))

    epoch_steps = len(train_dataloader)
    global_step = 0
    best_test_loss = np.inf
    for epoch in tqdm(range(cfg['epochs']), desc='Epoch', total=cfg['epochs'], position=0):
        model.train()
        for i, traj in tqdm(enumerate(train_dataloader), desc='Train Batch', position=1, total=len(train_dataloader), leave=False):
            train_start_time = time()
            traj = traj.to(device=cfg['device'], dtype=torch.float32)
            global_step = i + epoch * epoch_steps

            loss, metrics, info = model.get_loss(traj, cfg['train_dataset']['signal_length'], 0)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['optimizer']['grad_clip'] if 'grad_clip' in cfg['optimizer'] else np.inf)
            optimizer.step()
            metrics['grad_norm'] = grad_norm
            metrics['step_time'] = time() - train_start_time

            if global_step % cfg['log_every'] == 0:
                train_metrics = {f'train/{k}' : v for k, v in metrics.items()}
                for k, v in train_metrics.items():
                    logger.add_scalar(k, v, global_step)

            if global_step % cfg['viz_every'] == 0:
                visualize(info, logger, global_step, prefix='train')

        with torch.no_grad():
            model.eval()
            metric_list = []
            for i, traj in tqdm(enumerate(test_dataloader), desc='Test Batch', position=1, total=len(test_dataloader), leave=False):
                test_start_time = time()
                traj = traj.to(device=cfg['device'], dtype=torch.float32)
                loss, metrics, info = model.get_loss(traj, cfg['test_dataset']['signal_length'], 0)
                metrics['step_time'] = time() - test_start_time

            metric_list.append(metrics)
            metrics = {k : np.mean([m[k] for m in metric_list]) for k in metric_list[0].keys()}
            metrics = {f'test/{k}' : v for k, v in metrics.items()}
            for k, v in metrics.items():
                logger.add_scalar(k, v, global_step)

            if metrics['test/loss'] < best_test_loss:
                best_test_loss = metrics['test/loss']
                torch.save(model.state_dict(), os.path.join(cfg['log_dir'], cfg['name'] + timestring, 'best_model.pt'))

            visualize(info, logger, global_step, prefix='test')
            visualize_generations(model, logger, global_step, length=cfg['test_dataset']['signal_length'], prefix='test', device=cfg['device'])

def get_optimizer(cfg, model):
    if cfg['type'] == 'adam':
        return torch.optim.Adam(model.parameters(), **cfg['params'])
    else:
        raise NotImplementedError(f"Optimizer type {cfg['type']} not implemented")

def plt_prep(tensor):
    return tensor.detach().cpu().numpy().squeeze()

def visualize(info, logger, global_step, n_samples=3, prefix='train'):
    plot_fig = plt.figure(0)
    delta_t_fig = plt.figure(1)
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
        delta_t_ax.plot(plt_prep(info['mask_data'][i]), label='delta_t', c='r')

        if i == 0:
            plot_ax.legend()
            delta_t_ax.legend()

    logger.add_figure(prefix + '/reconstruction', plot_fig, global_step)
    logger.add_figure(prefix + '/delta_t', delta_t_fig, global_step)

    plot_fig.clf()
    delta_t_fig.clf()

def visualize_generations(model, logger, global_step, n_samples=3, length=128, device='cpu', prefix='test'):
    generated_trajs, boundary_data_list, _ = model.full_generation(torch.zeros(n_samples, 0, 1, device=device), seq_size=length)
    plot_fig = plt.figure(0)
    for i in range(n_samples):
        # Plot generated trajectories and discrete segmentation points
        plot_ax = plot_fig.add_subplot(n_samples, 1, i+1)
        plot_ax.plot(plt_prep(generated_trajs[i]), label='generated', zorder=10, c='g')
        segmentations = plt_prep(boundary_data_list[i])
        indices = np.arange(segmentations.shape[0])
        plot_ax.vlines(indices[segmentations == 1], np.min(plt_prep(generated_trajs[i])), np.max(plt_prep(generated_trajs[i])), label='segmentations', zorder=0, color='k')

        if i == 0:
            plot_ax.legend()

    logger.add_figure(prefix + '/generation', plot_fig, global_step)

    plot_fig.clf()

if __name__ == '__main__':
    test_full_prototype()