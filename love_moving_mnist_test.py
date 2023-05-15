import os
from datetime import datetime, timezone, timedelta
from time import time

import hydra
from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import StochasticMovingMNIST as Dataset
from love.love_hssm import EnvModel
from utils import make_scheduler

@hydra.main(version_base='1.3', config_path='cfgs', config_name='love_moving_mnist_experiment')
def test_full_prototype(cfg):
    np.random.seed(cfg['np_seed'])
    train_dataset = Dataset(**cfg['train_dataset'])
    train_dataloader = DataLoader(train_dataset, **cfg['dataloader'])

    test_dataset = Dataset(**cfg['test_dataset'])
    test_dataloader = DataLoader(test_dataset, **cfg['dataloader'])

    model = EnvModel(**cfg['model'])
    model.to(cfg['device'])

    if cfg['model_load_path'] is not None:
        print("MODEL LOADED")
        model.load_state_dict(torch.load(cfg['model_load_path'], map_location=cfg['device']))

    optimizer = get_optimizer(cfg['optimizer'], model)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1)
    timestring = datetime.now(tz=timezone(timedelta(hours=-5))).strftime("_%m-%d-%Y_%H-%M-%S") # EST, No daylight savings
    logger = SummaryWriter(os.path.join(cfg['log_dir'], cfg['name'] + timestring))
    logger.add_text('config', str(cfg))
    logger.add_text('model', str(model))

    epoch_steps = len(train_dataloader)
    global_step = 0
    best_test_loss = np.inf
    for epoch in tqdm(range(cfg['epochs']), desc='Epoch', total=cfg['epochs'], position=0):
        model.train()
        for i, (frames, context, frame_coords, context_coords) in tqdm(enumerate(train_dataloader), desc='Train Batch', position=1, total=len(train_dataloader), leave=False):
            train_start_time = time()
            context = context.to(device=cfg['device'], dtype=torch.float32)
            frames = frames.to(device=cfg['device'], dtype=torch.float32)
            global_step = i + epoch * epoch_steps

            optimizer.zero_grad()
            loss, metrics, info = model.get_loss(torch.cat([context, frames], dim=1), frames.shape[1], context.shape[1])
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['optimizer']['grad_clip'] if 'grad_clip' in cfg['optimizer'] else np.inf)
            optimizer.step()
            lr_scheduler.step()
            metrics['grad_norm'] = grad_norm
            metrics['lr'] = lr_scheduler.get_last_lr()[0]
            metrics['step_time'] = time() - train_start_time

            if global_step % cfg['log_every'] == 0:
                train_metrics = {f'train/{k}' : v for k, v in metrics.items()}
                for k, v in train_metrics.items():
                    logger.add_scalar(k, v, global_step)

            if global_step % cfg['viz_every'] == 0:
                visualize(info, frame_coords, logger, global_step, prefix='train')
                visualize_generations(model, context, logger, global_step, prefix='train')

        with torch.no_grad():
            model.eval()
            metric_list = []
            for i, (frames, context, frame_coords, context_coords) in tqdm(enumerate(test_dataloader), desc='Test Batch', position=1, total=len(test_dataloader), leave=False):
                test_start_time = time()
                frames = frames.to(device=cfg['device'], dtype=torch.float32)
                context = context.to(device=cfg['device'], dtype=torch.float32)
                loss, metrics, info = model.get_loss(torch.cat([context, frames], dim=1), frames.shape[1], context.shape[1])
                metrics['step_time'] = time() - test_start_time

            metric_list.append(metrics)
            metrics = {k : np.mean([m[k] for m in metric_list]) for k in metric_list[0].keys()}
            metrics = {f'test/{k}' : v for k, v in metrics.items()}
            for k, v in metrics.items():
                logger.add_scalar(k, v, global_step)

            if metrics['test/loss'] < best_test_loss:
                best_test_loss = metrics['test/loss']
                torch.save(model.state_dict(), os.path.join(cfg['log_dir'], cfg['name'] + timestring, 'best_model.pt'))
                torch.save(model, os.path.join(cfg['log_dir'], cfg['name'] + timestring, 'best_full_model.pt'))

            visualize(info, frame_coords, logger, global_step, prefix='test')
            visualize_generations(model, context, logger, global_step, prefix='test')


            torch.save(model.state_dict(), os.path.join(cfg['log_dir'], cfg['name'] + timestring, 'latest_model.pt'))
            torch.save(model, os.path.join(cfg['log_dir'], cfg['name'] + timestring, 'latest_full_model.pt'))

def get_optimizer(cfg, model):
    if cfg['type'] == 'adam':
        return torch.optim.Adam(model.parameters(), **cfg['params'])
    else:
        raise NotImplementedError(f"Optimizer type {cfg['type']} not implemented")

def plt_prep(tensor):
    return tensor.detach().cpu().numpy().squeeze()

COLORS = ['r', 'y', 'b', 'c']

def visualize(info, frame_coords, logger, global_step, n_samples=3, prefix='train'):
    plot_fig = plt.figure(0)
    delta_t_fig = plt.figure(1)
    delta_t_logit_fig = plt.figure(2)
    for i in range(n_samples):
        segmentations = plt_prep(info['mask_data'][i])
        indices = np.arange(segmentations.shape[0])
        plot_ax = plot_fig.add_subplot(n_samples, 1, i+1)
        max_y = -np.inf
        min_y = np.inf
        for j in range(frame_coords.shape[-1]):
            traj = plt_prep(frame_coords[i, :, j])
            plot_ax.plot(traj, c=COLORS[j], label=f'digit_{j // 2}_x' if j % 2 == 0 else f'digit_{j // 2}_y', zorder=10)
            max_y = np.maximum(max_y, np.max(traj))
            min_y = np.minimum(min_y, np.min(traj))
        plot_ax.vlines(indices[segmentations > 0.5], min_y, max_y, color='k', label='segmentation', zorder=0)

        # Plot delta_t for sequence
        delta_t_ax = delta_t_fig.add_subplot(n_samples, 1, i+1)
        delta_t_ax.plot(plt_prep(info['mask_data'][i]), label='delta_t', c='r')

        # # Plot delta_t logit for sequence
        # delta_t_logit_ax = delta_t_logit_fig.add_subplot(n_samples, 1, i+1)
        # delta_t_logit_ax.plot(plt_prep(torch.sigmoid(info['segmentation_post_logits'][i, :, 0])), label='post_prob', c='b')
        # delta_t_logit_ax.plot(plt_prep(torch.sigmoid(info['segmentation_prior_logits'][i, :, 0])), label='prior prob', c='r')

        if i == 0:
            delta_t_ax.legend()
            # delta_t_logit_ax.legend()
            plot_ax.legend()

    logger.add_figure(prefix + '/reconstruction', plot_fig, global_step)
    logger.add_figure(prefix + '/delta_t', delta_t_fig, global_step)
    logger.add_figure(prefix + '/delta_t_logit', delta_t_logit_fig, global_step)

    logger.add_video(prefix + '/ground_truth', info['ground_truth_traj'][0:1], global_step)
    logger.add_video(prefix + '/reconstructed', torch.clamp(info['reconstructed_traj'][0:1], 0, 1), global_step)

    plot_fig.clf()
    delta_t_fig.clf()
    delta_t_logit_fig.clf()

def visualize_generations(model, context, logger, global_step, n_samples=3, prefix='test'):
    generated_trajs, boundary_data_list, _ = model.full_generation(context, seq_size=64)
    segmentation_prob_fig = plt.figure(1)
    for i in range(n_samples):
        segmentations = plt_prep(boundary_data_list[i])
        segmentation_prob_ax = segmentation_prob_fig.add_subplot(n_samples, 1, i+1)
        segmentation_prob_ax.plot(segmentations, label='segmentation probability', c='b')

        if i == 0:
            segmentation_prob_ax.legend()

    logger.add_figure(prefix + '/generation_segmentation_prob', segmentation_prob_fig, global_step)
    logger.add_video(prefix + '/generated', torch.clamp(generated_trajs[0:1], 0, 1), global_step)
    logger.add_video(prefix + '/generated_context', torch.clamp(context[0:1], 0, 1), global_step)

    segmentation_prob_fig.clf()

if __name__ == '__main__':
    test_full_prototype()