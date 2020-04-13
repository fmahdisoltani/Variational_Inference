import torch
import numpy as np
from torch.autograd import grad


def nth_derivative(f, wrt, n):
    for i in range(n):
        grads = grad(f, wrt, create_graph=True)[0]
        f = grads.sum()
    return grads


def log_normalize(x):
    return x - torch.logsumexp(x, 0)


def analytic_posterior(x, t):
    empirical_mean = torch.sum(x) / len(x)
    empirical_var = torch.sum((x-empirical_mean) ** 2) / len(x)

    likelihood_s = torch.tensor(5.5)
    analytical_posterior_var = ((1 / likelihood_s ** 2) * X.T @ X + 1) ** -1
    analytical_posterior_mean = analytical_posterior_var * (0.9 + ((1 / 5.5 ** 2) * X.T @ T))