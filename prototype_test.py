import os
from datetime import datetime, timezone, timedelta

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import ComplicatedSinusoid
from models import PrototypeModel
from utils import LinearScheduler, LogarithmicScheduler

def test_prototype(cfg):
    np.random.seed(cfg['np_seed'])
    train_dataset = ComplicatedSinusoid(**cfg['train_dataset'])
    train_dataloader = DataLoader(train_dataset, **cfg['dataloader'])

    test_dataset = ComplicatedSinusoid(**cfg['test_dataset'])
    test_dataloader = DataLoader(test_dataset, **cfg['dataloader'])

    model = PrototypeModel(data_dim=1, seq_len=cfg['train_dataset']['signal_length'], **cfg['model'])
    model.to(cfg['device'])
    optimizer = torch.optim.Adam(model.parameters(), **cfg['optimizer'])
    temp_scheduler = LogarithmicScheduler(**cfg['temp_scheduler'])
    time_loss_weight_scheduler = LinearScheduler(**cfg['time_loss_weight_scheduler'])

    timestring = datetime.now(tz=timezone(timedelta(hours=-5))).strftime("_%m-%d-%Y_%H-%M-%S") # EST, No daylight savings
    logger = SummaryWriter(os.path.join(cfg['log_dir'], cfg['name'] + timestring))
    logger.add_text('config', str(cfg))
    logger.add_text('model', str(model))

    epoch_steps = len(train_dataloader)
    global_step = 0
    for epoch in tqdm(range(cfg['epochs']), desc='Epoch', total=cfg['epochs'], position=0):
        temp = temp_scheduler.get_value(epoch)
        model.set_temperature(temp)
        logger.add_scalar('train/temp', temp, global_step)
        time_loss_weight = time_loss_weight_scheduler.get_value(epoch)
        model.set_time_loss_weight(time_loss_weight)
        logger.add_scalar('train/time_loss_weight', time_loss_weight, global_step)
        model.train()
        for i, traj in tqdm(enumerate(train_dataloader), desc='Train Batch', position=1, total=len(train_dataloader), leave=False):
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
                visualize(info, logger, global_step, prefix='train')

        with torch.no_grad():
            model.eval()
            metric_list = []
            for i, traj in tqdm(enumerate(test_dataloader), desc='Test Batch', position=1, total=len(test_dataloader), leave=False):
                traj = traj.to(device=cfg['device'], dtype=torch.float32)
                loss, metrics, info = model.get_loss(traj)

            metric_list.append(metrics)
            metrics = {k : np.mean([m[k] for m in metric_list]) for k in metric_list[0].keys()}
            metrics = {f'test/{k}' : v for k, v in metrics.items()}
            for k, v in metrics.items():
                logger.add_scalar(k, v, global_step)

            visualize(info, logger, global_step, prefix='test')

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

    logger.add_figure(prefix + '/reconstruction', plot_fig, global_step)
    logger.add_figure(prefix + '/delta_t', delta_t_fig, global_step)
    logger.add_figure(prefix + '/delta_t_logit', delta_t_logit_fig, global_step)
    logger.add_figure(prefix + '/latent_features', latent_features_fig, global_step)

    plot_fig.clf()
    delta_t_fig.clf()
    delta_t_logit_fig.clf()
    latent_features_fig.clf()

if __name__ == '__main__':
    cfg = dict(
        log_dir='logs/prototype_with_eval',
        name='test_complicated_sinusoid',
        device='cuda:0',
        log_every=50,
        viz_every=100,
        np_seed=0,
        epochs=1500,
        grad_clip=50.0,
        model=dict(
            latent_dim=4,
            positional_encoding_dim=64,
            transformer_encoder_layers=2,
            max_subseq_len=257,
            reconstruction_loss_weight=1.0,
            time_gradient_scalar=1.0,
        ),
        optimizer=dict(
            lr=3e-4,
            weight_decay=1e-5,
        ),
        temp_scheduler=dict(
            start_value=2.0,
            end_value=0.5,
            start_step=150,
            end_step=400,
        ),
        time_loss_weight_scheduler=dict(
            start_value=0.0,
            end_value=0.20,
            start_step=10,
            end_step=10,
        ),
        train_dataset=dict(
            dataset_size=10000,
            signal_length=128,
        ),
        test_dataset=dict(
            dataset_size=1000,
            signal_length=128,
        ),
        dataloader=dict(
            batch_size=64,
            shuffle=False,
            num_workers=0,
        ),
    )
    test_prototype(cfg)