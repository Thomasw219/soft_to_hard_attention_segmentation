import os
from datetime import datetime, timezone, timedelta

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import FixedSizePiecewiseSine
from models import PrototypeModel
from utils import LinearScheduler

def test_prototype(cfg):
    np.random.seed(cfg['np_seed'])
    dataset = FixedSizePiecewiseSine(**cfg['dataset'])
    dataloader = DataLoader(dataset, **cfg['dataloader'])

    model = PrototypeModel(data_dim=1, seq_len=cfg['dataset']['signal_length'], **cfg['model'])
    model.to(cfg['device'])
    optimizer = torch.optim.Adam(model.parameters(), **cfg['optimizer'])
    temp_scheduler = LinearScheduler(**cfg['temp_scheduler'])

    timestring = datetime.now(tz=timezone(timedelta(hours=-5))).strftime("_%m-%d-%Y_%H-%M-%S") # EST, No daylight savings
    logger = SummaryWriter(os.path.join(cfg['log_dir'], cfg['name'] + timestring))
    logger.add_text('config', str(cfg))
    logger.add_text('model', str(model))

    epoch_steps = len(dataloader)
    for epoch in tqdm(range(cfg['epochs']), desc='Epoch', total=cfg['epochs'], position=0):
        temp = temp_scheduler.get_value(epoch)
        model.set_temperature(temp)
        logger.add_scalar('train/temp', temp, epoch)
        for i, traj in tqdm(enumerate(dataloader), desc='Batch', position=1, total=len(dataloader), leave=False):
            traj = traj.to(device=cfg['device'], dtype=torch.float32)
            global_step = i + epoch * epoch_steps

            optimizer.zero_grad()
            loss, metrics, info = model.get_loss(traj)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            metrics['grad_norm'] = grad_norm
            optimizer.step()

            if global_step % cfg['log_every'] == 0:
                train_metrics = {f'train/{k}' : v for k, v in metrics.items()}
                for k, v in train_metrics.items():
                    logger.add_scalar(k, v, global_step)

            if global_step % cfg['viz_every'] == 0:
                visualize(info, logger, global_step)

def plt_prep(tensor):
    return tensor.detach().cpu().numpy().squeeze()

def visualize(info, logger, global_step, n_samples=3):
    plot_fig = plt.figure(0)
    delta_t_fig = plt.figure(1)
    delta_t_logit_fig = plt.figure(2)
    latent_features_fig = plt.figure(3)
    for i in range(n_samples):
        # Plot ground truth and reconstruction for n_samples
        plot_ax = plot_fig.add_subplot(n_samples, 1, i+1)
        plot_ax.plot(plt_prep(info['ground_truth_traj'][i]), label='ground truth', c='b')
        plot_ax.plot(plt_prep(info['reconstructed_traj'][i]), label='reconstruction', c='g')

        # Plot delta_t for sequence
        delta_t_ax = delta_t_fig.add_subplot(n_samples, 1, i+1)
        delta_t_ax.plot(plt_prep(info['delta_t'][i]), label='delta_t', c='r')

        # Plot delta_t logit for sequence
        delta_t_logit_ax = delta_t_logit_fig.add_subplot(n_samples, 1, i+1)
        delta_t_logit_ax.plot(plt_prep(info['delta_t_logits'][i, :, 0]), label='0 logits', c='b')
        delta_t_logit_ax.plot(plt_prep(info['delta_t_logits'][i, :, 1]), label='1 logits', c='g')

        # Plot latent features for sequence
        latent_features_ax = latent_features_fig.add_subplot(n_samples, 1, i+1)
        for j in range(info['latent_feats'].shape[-1]):
            latent_features_ax.plot(plt_prep(info['latent_feats'][i, :, j]), label=f'latent feature {j}')

        if i == 0:
            plot_ax.legend()
            delta_t_ax.legend()
            delta_t_logit_ax.legend()

    logger.add_figure('reconstruction', plot_fig, global_step)
    logger.add_figure('delta_t', delta_t_fig, global_step)
    logger.add_figure('delta_t_logit', delta_t_logit_fig, global_step)
    logger.add_figure('latent_features', latent_features_fig, global_step)

    plot_fig.clf()
    delta_t_fig.clf()
    delta_t_logit_fig.clf()
    latent_features_fig.clf()

if __name__ == '__main__':
    cfg = dict(
        log_dir='logs/prototype_temp_anneal',
        name='test',
        device='cuda:0',
        log_every=50,
        viz_every=100,
        np_seed=0,
        epochs=100,
        grad_clip=50.0,
        model=dict(
            latent_dim=4,
            max_subseq_len=257,
            reconstruction_loss_weight=1.0,
            time_loss_weight=0.5,
            time_gradient_scalar=0.01,
        ),
        optimizer=dict(
            lr=1e-4,
            weight_decay=1e-5,
        ),
        temp_scheduler=dict(
            start_value=1.0,
            end_value=0.01,
            start_step=25,
            end_step=75,
        ),
        dataset=dict(
            dataset_size=10000,
            piece_length=20,
            signal_length=128,
        ),
        dataloader=dict(
            batch_size=64,
            shuffle=False,
            num_workers=0,
        ),
    )
    test_prototype(cfg)