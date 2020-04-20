import numpy as np
import torch
from sampling import gaussian_sample
from utils import log_normalize
from sklearn.datasets.samples_generator import make_blobs


def create_unimodal_data(num_samples, w, scale, low=-10, high=10):
    x = np.random.uniform(low=low, high=high, size=(num_samples, 1))
    t = w * x + np.random.normal(size=(num_samples, 1), scale=scale)
    x = torch.tensor(x)
    t = torch.tensor(t)
    return x, t


def create_bimodal_data(bias, num_samples, scale_1, scale_2, w, pai,  low=-50, high=50):
    # generate data:
    # p(t|w, x) = a * N(t; wx, scale_1) + (1-a) * N(t; (w+5)x, scale_2)
    #         (1) sample x_i
    #         (2) compute y_i from GMM

    x = np.random.uniform(low=low, high=high, size=(num_samples, 1))
    # samples, idx = gmm_sample_full(torch.tensor([0., 0.]), torch.tensor([scale_1, scale_2]), torch.tensor([pai, 1-pai]), num_samples)

    samples = torch.cat([gaussian_sample(mean, std, num_samples)[:, np.newaxis, :]
                         for mean, std in zip(torch.tensor([0., 0.]),  torch.tensor([scale_1, scale_2]))], axis=1)

    samples[:, 0, :] += torch.tensor(w * x)
    samples[:, 1, :] += torch.tensor((w + bias) * x)
    weights = torch.tensor([pai, 1-pai])
    ixs = torch.multinomial(weights, num_samples, replacement=True)
    t = torch.stack([samples[i, ix, :] for i, ix in enumerate(ixs)])
    # import matplotlib.pyplot as plt
    # plt.scatter(x, t)
    return x, t


def create_bimodal_data_fix(bias, num_samples):
    x = torch.ones([num_samples, 1])
    t = bias * torch.ones([num_samples, 1])
    return x, t