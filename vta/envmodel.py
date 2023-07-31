from vta.modules import *
from vta.utils import *
from vta.hssm import HierarchicalStateSpaceModel


class EnvModel(nn.Module):
    def __init__(self,
                 action_encoder,
                 encoder,
                 decoder,
                 belief_size,
                 state_size,
                 num_layers,
                 max_seg_len,
                 max_seg_num,
                 rec_coeff=1.0,
                 abs_state_coeff=1.0,
                 obs_state_coeff=1.0,
                 mask_coeff=1.0,
                 compression_coeff=1.0,
                 ):
        super(EnvModel, self).__init__()
        ################
        # network size #
        ################
        self.belief_size = belief_size
        self.state_size = state_size
        self.num_layers = num_layers
        self.max_seg_len = max_seg_len
        self.max_seg_num = max_seg_num

        self.rec_coeff = rec_coeff
        self.abs_state_coeff = abs_state_coeff
        self.obs_state_coeff = obs_state_coeff
        self.mask_coeff = mask_coeff
        self.compression_coeff = compression_coeff

        ###############
        # init models #
        ###############
        # state space model
        self.state_model = HierarchicalStateSpaceModel(action_encoder=action_encoder,
                                                       encoder=encoder,
                                                       decoder=decoder,
                                                       belief_size=self.belief_size,
                                                       state_size=self.state_size,
                                                       num_layers=self.num_layers,
                                                       max_seg_len=self.max_seg_len,
                                                       max_seg_num=self.max_seg_num)

    def forward(self, obs_data_list, obs_points_list, seq_size, init_size, obs_std=1.0):
        ############################
        # (1) run over state model #
        ############################
        [obs_rec_list,
         prior_boundary_log_density_list,
         post_boundary_log_density_list,
         prior_abs_state_list,
         post_abs_state_list,
         prior_obs_state_list,
         post_obs_state_list,
         boundary_data_list,
         prior_boundary_list,
         post_boundary_list] = self.state_model(obs_data_list, seq_size, init_size)

        ########################################################
        # (2) compute obs_cost (sum over spatial and channels) #
        ########################################################
        obs_target_list = obs_data_list[:, init_size:-init_size]
        obs_cost = - Normal(obs_rec_list, obs_std).log_prob(obs_target_list)
        obs_cost = obs_cost.sum(dim=[2, 3, 4])

        #######################
        # (3) compute kl_cost #
        #######################
        # compute kl related to states
        kl_abs_state_list = []
        kl_obs_state_list = []
        for t in range(seq_size):
            # read flag
            read_data = boundary_data_list[:, t].detach()

            # kl divergences (sum over dimension)
            kl_abs_state = kl_divergence(post_abs_state_list[t], prior_abs_state_list[t]) * read_data
            kl_obs_state = kl_divergence(post_obs_state_list[t], prior_obs_state_list[t])
            kl_abs_state_list.append(kl_abs_state.sum(-1))
            kl_obs_state_list.append(kl_obs_state.sum(-1))
            
        kl_abs_state_list = torch.stack(kl_abs_state_list, dim=1)
        kl_obs_state_list = torch.stack(kl_obs_state_list, dim=1)

        # compute kl related to boundary
        kl_mask_list = (post_boundary_log_density_list - prior_boundary_log_density_list)

        obs_points_list = obs_points_list[:, init_size:-init_size]
        # return
        return {'rec_data': obs_rec_list,
                'mask_data': boundary_data_list,
                'mask_data_true': obs_points_list,
                'obs_cost': obs_cost,
                'kl_abs_state': kl_abs_state_list,
                'kl_obs_state': kl_obs_state_list,
                'kl_mask': kl_mask_list,
                'p_mask': prior_boundary_list.mean,
                'q_mask': post_boundary_list.mean,
                'p_ent': prior_boundary_list.entropy(),
                'q_ent': post_boundary_list.entropy(),
                'beta': self.state_model.mask_beta,
                'train_loss': obs_cost.mean() + kl_abs_state_list.mean() + kl_obs_state_list.mean() + kl_mask_list.mean()}

    def get_loss(self, obs_data_list, action_list):
        init_size = 1
        seq_size = obs_data_list.shape[1] - 2 * init_size

        [
            action_rec_mean,
            action_rec_std,
            prior_boundary_log_density_list,
            post_boundary_log_density_list,
            prior_abs_state_list,
            post_abs_state_list,
            prior_obs_state_list,
            post_obs_state_list,
            boundary_data_list,
            prior_boundary_list,
            post_boundary_list,
        ] = self.state_model(obs_data_list, action_list, seq_size, init_size)

        ########################################################
        # (2) compute obs_cost (sum over spatial and channels) #
        ########################################################
        action_dist = torch.distributions.Normal(
            action_rec_mean.reshape(-1, action_rec_mean.shape[-1]),
            torch.nn.functional.softplus(action_rec_std.reshape(-1, action_rec_std.shape[-1])),
        )
        action_costs = action_dist.log_prob(action_list[:, init_size:-init_size].reshape(-1, action_list.shape[-1])).sum(-1)
        action_cost = -torch.mean(action_costs)

        #######################
        # (3) compute kl_cost #
        #######################
        # compute kl related to states
        kl_abs_state_list = []
        kl_obs_state_list = []
        for t in range(seq_size):
            # read flag
            read_data = boundary_data_list[:, t].detach()

            # kl divergences (sum over dimension)
            kl_abs_state = kl_divergence(post_abs_state_list[t], prior_abs_state_list[t]) * read_data
            kl_obs_state = kl_divergence(post_obs_state_list[t], prior_obs_state_list[t])
            kl_abs_state_list.append(kl_abs_state.sum(-1))
            kl_obs_state_list.append(kl_obs_state.sum(-1))

        kl_abs_state_list = torch.stack(kl_abs_state_list, dim=1)
        kl_obs_state_list = torch.stack(kl_obs_state_list, dim=1)

        # compute kl related to boundary
        kl_mask_list = (post_boundary_log_density_list - prior_boundary_log_density_list)

        # compression loss
        compression_loss = torch.mean(post_boundary_list.probs)

        loss = (
            self.rec_coeff * action_cost.mean()
            + self.abs_state_coeff * kl_abs_state_list.mean()
            + self.obs_state_coeff * kl_obs_state_list.mean()
            + self.mask_coeff * kl_mask_list.mean()
            + self.compression_coeff * compression_loss
        )

        metrics = {
            'loss': loss,
            'reconstruction_loss': action_cost.mean(),
            'state_kl_loss': kl_obs_state_list.mean(),
            'abs_kl_loss': kl_abs_state_list.mean(),
            'mask_kl_loss': kl_mask_list.mean(),
            'compression_rate': 1 / torch.max(torch.mean(boundary_data_list), torch.tensor(1 / seq_size, device=boundary_data_list.device)),
            'compression_loss': compression_loss,
        }

        info = {
            'ground_truth_obs' : obs_data_list[:, init_size:-init_size],
            'ground_truth_act' : action_list[:, init_size:-init_size],
            'reconstructed_act' : action_rec_mean,
            'segmentation_prior_probs' : prior_boundary_list.probs,
            'segmentation_post_probs' : post_boundary_list.probs,
            'segmentation_samples' : boundary_data_list.squeeze(-1),
        }

        return loss, metrics, info