import torch

def sample_binary_concrete(logits, temp, hard=False, eps=1e-7):
    y = (logits + torch.log(torch.rand_like(logits) + eps) - torch.log(1 - torch.rand_like(logits) + eps)) / temp
    concrete = torch.sigmoid(y)
    if hard:
        concrete = torch.round(concrete)
    return concrete, y

def y_log_density(y, logits, temp):
    return torch.log(temp) - temp * y + logits - 2 * torch.log(1 + torch.exp(-temp * y + logits))

def y_kl_divergence(y_samples, prior_logits, prior_temp, posterior_logits, posterior_temp, kl_balance=0.5):
    prior_log_density = y_log_density(y_samples, prior_logits, prior_temp)
    posterior_log_density = y_log_density(y_samples, posterior_logits, posterior_temp)

    prior_kl = posterior_log_density.detach() - prior_log_density
    post_kl = posterior_log_density - prior_log_density.detach()
    kl = kl_balance * prior_kl + (1 - kl_balance) * post_kl
    return kl
