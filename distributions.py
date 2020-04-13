import torch
import math
import torch.nn as nn
import numpy as np
from utils import log_normalize


def gaussian(x, mean, std):
    return (1 / torch.sqrt(torch.FloatTensor([2*math.pi])*std**2)) * torch.exp(-((x - mean) ** 2.) / (2 * std**2))


def log_gaussian(x, mean, std):
    return -0.5 * torch.log(2 * np.pi * std ** 2) - (0.5 * (1 / (std ** 2)) * (x - mean) ** 2)


def gmm(x, means, variances, pais):
    return sum([pai * gaussian(x, mu, var) for (pai, mu, var) in zip(pais, means, variances)])


def log_gmm(x, means, stds, log_pais):
    component_log_densities = torch.stack([log_gaussian(x, mu, std) for (mu, std) in zip(means, stds)]).T
    # log_weights = torch.log(pais)
    log_weights = log_normalize(log_pais)
    return torch.logsumexp(component_log_densities + log_weights, axis=-1, keepdims=False)


class TwoMode(nn.Module):
    # negative log likelihood is in this form
    def __init__(self):
        super().__init__()
        self.function = lambda x: x ** 2 + torch.exp(-2 / (10 * (x - 1)) ** 2) - 2

    def forward(self, x):
        return self.function(x)


class GMM(nn.Module):
    def __init__(self, num_components, means, variances, pais):
        super().__init__()
        self.num_components = num_components
        self.means = torch.tensor(means, requires_grad=True)
        self.variances = torch.tensor(variances, requires_grad=True)
        self.pais = torch.tensor(pais, requires_grad=True)
        self.function = lambda x: gmm(x, self.means, self.variances, self.pais)

    def forward(self, x):
        return self.function(x)


class NegLogGMM(GMM):
    def __init__(self, num_components, means, variances, pais):
        super().__init__(num_components, means, variances, pais)

    def forward(self, x):
        return -torch.log(self.function(x))


class Gaussian(nn.Module):
    def __init__(self, mean, sigma):
        super().__init__()
        self.mean = mean
        self.sigma = sigma
        self.function = lambda x: gaussian(x, self.mean, self.sigma)

    def forward(self, x):
        return self.function(x)


class LogGaussian(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        return log_gaussian(x, self.mean, self.std)


class Uniform(nn.Module):
    def __init__(self, u, l):
        super().__init__()
        self.up = u
        self.low = l
        self.function = lambda x: torch.tensor(1. / (u - l))

    def forward(self, x):
        return self.function(x)


if __name__ == '__main__':
    x = torch.FloatTensor([2])
    a = gaussian(x, torch.FloatTensor([0.]), torch.FloatTensor([1.]))