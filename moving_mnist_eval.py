import torch
import numpy as np

from data import StochasticMovingMNIST as Dataset

MODEL_DIR = 'logs/frozen_seg_post/frozen_seg_post_04-22-2023_23-15-50/'
DEVICE = torch.device('cuda:0')

torch.manual_seed(0)
np.random.seed(0)

def main():
    model = torch.load(MODEL_DIR + 'best_full_model.pt', map_location=DEVICE)

    test_dataset = Dataset(False, num_digits=1, deterministic=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

    for i, (frames, context, frame_coords, context_coords) in enumerate(test_dataloader):
        if i == 1:
            break

        frames = frames.to(device=DEVICE, dtype=torch.float32)
        context = context.to(device=DEVICE, dtype=torch.float32)

        # _, info = model.forward(context, frames)
        # gt_segmentations = info['segmentation_samples']

        # generated_traj, info = model.generate(context)

        # generated_segmentations = info['segmentation_samples']

        # # print(gt_segmentations)
        # # print(generated_segmentations)

        # np_gt = frames.detach().cpu().numpy()
        # np_gen = generated_traj.detach().cpu().numpy()

        # np_gt_seg = gt_segmentations.detach().cpu().numpy()
        # np_gen_seg = generated_segmentations.detach().cpu().numpy()


        # print(np_gen.shape)
        # print(np_gen_seg)

        # save_segmentations(np_gt, np_gt_seg, "actual", i)
        # save_segmentations(np_gen, np_gen_seg, "generated", i)

        abstract_eps_0_0 = torch.randn(1, 64, model.cfg.abstract_rep_stoch_dim).to(device=DEVICE, dtype=torch.float32)
        state_eps = torch.randn(1, 64, model.cfg.state_rep_stoch_dim).to(device=DEVICE, dtype=torch.float32)
        generated_traj_0_0, info = model.generate(context, given_abstract_eps=abstract_eps_0_0, given_state_eps=state_eps)
        segmentations_0_0 = info['segmentation_samples']

        print(segmentations_0_0)

        abstract_eps_1_0 = abstract_eps_0_0.clone()
        abstract_eps_1_0[0, 3] = torch.randn_like(abstract_eps_1_0[0, 3])
        generated_traj_1_0, info = model.generate(context, given_abstract_eps=abstract_eps_1_0, given_state_eps=state_eps)
        segmentations_1_0 = info['segmentation_samples']

        print(segmentations_1_0)

        abstract_eps_0_1 = abstract_eps_0_0.clone()
        abstract_eps_0_1[0, 7] = torch.randn_like(abstract_eps_0_1[0, 7])
        generated_traj_0_1, info = model.generate(context, given_abstract_eps=abstract_eps_0_1, given_state_eps=state_eps)
        segmentations_0_1 = info['segmentation_samples']

        print(segmentations_0_1)

        abstract_eps_1_1 = abstract_eps_1_0.clone()
        abstract_eps_1_1[0, 11] = torch.randn_like(abstract_eps_1_1[0, 11])
        generated_traj_1_1, info = model.generate(context, given_abstract_eps=abstract_eps_1_1, given_state_eps=state_eps)
        segmentations_1_1 = info['segmentation_samples']

        print(segmentations_1_1)

        save_segmentations(context.detach().cpu().numpy(), np.zeros_like(segmentations_0_0.detach().cpu().numpy())[:, :context.shape[1]], "context", i)
        save_segmentations(generated_traj_0_0.detach().cpu().numpy(), segmentations_0_0.detach().cpu().numpy(), "0_0", i)
        save_segmentations(generated_traj_1_0.detach().cpu().numpy(), segmentations_1_0.detach().cpu().numpy(), "1_0", i)
        save_segmentations(generated_traj_0_1.detach().cpu().numpy(), segmentations_0_1.detach().cpu().numpy(), "0_1", i)
        save_segmentations(generated_traj_1_1.detach().cpu().numpy(), segmentations_1_1.detach().cpu().numpy(), "1_1", i)

def save_segmentations(frames, segmentations, name, i):
    import matplotlib.pyplot as plt

    segment_image = frames[0][0][0].copy()
    seg_counter = 0
    for j, seg in enumerate(segmentations[0][1:]):
        if seg == 1:
            plt.imsave(f"media/mnist/{name}{i}_{seg_counter}.png", segment_image, cmap='gray')
            seg_counter += 1
            segment_image = frames[0][j+1][0].copy()
        else:
            segment_image = np.concatenate((segment_image, np.ones((64, 1))), axis=1)
            segment_image = np.concatenate((segment_image, frames[0][j+1][0]), axis=1)
    plt.imsave(f"media/mnist/{name}{i}_{seg_counter}.png", segment_image, cmap='gray')

if __name__ == '__main__':
    main()