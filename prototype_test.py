import os
from datetime import datetime, timezone, timedelta

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import FixedSizePiecewiseSine
from models import PrototypeModel

def test_prototype(cfg):
    np.random.seed(cfg['np_seed'])
    dataset = FixedSizePiecewiseSine(**cfg['dataset'])
    dataloader = DataLoader(dataset, **cfg['dataloader'])

    model = PrototypeModel(data_dim=1, seq_len=cfg['dataset']['signal_length'], **cfg['model'])
    model.to(cfg['device'])
    optimizer = torch.optim.Adam(model.parameters(), **cfg['optimizer'])

    timestring = datetime.now(tz=timezone(timedelta(hours=-5))).strftime("_%m-%d-%Y_%H-%M-%S") # EST, No daylight savings
    logger = SummaryWriter(os.path.join(cfg['log_dir'], cfg['name'] + timestring))
    logger.add_text('config', str(cfg))
    logger.add_text('model', str(model))

    epoch_steps = len(dataset)
    for epoch in tqdm(range(cfg['epochs']), desc='Epoch', total=cfg['epochs'], position=0):
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
                pass

if __name__ == '__main__':
    cfg = dict(
        log_dir='logs/prototype',
        name='test',
        device='cuda:0',
        log_every=50,
        viz_every=100,
        np_seed=0,
        epochs=100,
        grad_clip=50.0,
        model=dict(
            latent_dim=4,
            max_subseq_len=129,
            reconstruction_loss_weight=1.0,
            time_loss_weight=0.0,
        ),
        optimizer=dict(
            lr=1e-3,
            weight_decay=1e-5,
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