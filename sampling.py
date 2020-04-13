import torch
from utils import log_normalize
import numpy as np

def gmm_sample(means, stds, log_pais, num_samples): #TODO: change std's to log_std's
    D = 1 # TODO: D is dimension, set it properly for multi-dimensional
    samples = torch.cat([gaussian_sample(mean, std, num_samples)[:, np.newaxis, :]
                         for mean, std in zip(means, stds)], axis=1)
    # ixs = np.random.choice(k, size=num_samples, p=np.exp(log_weights))
    # weights = log_normalize(log_pais)
    # log_weights = log_normalize(log_pais)
    # print(torch.exp(log_weights))
    # print(log_weights)
    weights = torch.exp(log_normalize(log_pais))
    ixs = torch.multinomial(weights, num_samples, replacement=True)
    # ixs = np.random.choice(2, size=num_samples, p=weights.detach())

    return torch.stack([samples[i, ix, :] for i, ix in enumerate(ixs)])


def gaussian_sample(mean, std, num_samples):
    D = 1 #np.shape(mean)[0]
    # ss = torch.randn(num_samples, D) * torch.exp(log_std) + mean
    # return ss

    eps = torch.randn(size=(num_samples, D))
    z = eps * std + mean
    return z