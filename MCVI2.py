import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm
from dataset import create_unimodal_data
from distributions import log_gaussian, log_gmm
from sampling import gmm_sample

plt.style.use('ggplot')


class MCElbo(torch.nn.Module):
    def __init__(self):
        super(MCElbo, self).__init__()
        self.n_latent = 100  # Number of latent samples
        self.softplus = torch.nn.Softplus()

        # The parameters we adjust during training.
        self.qm = torch.nn.Parameter(torch.randn(1, 1), requires_grad=True)
        self.qs = torch.nn.Parameter(torch.randn(1, 1), requires_grad=True)

        # create holders for prior mean and std, and likelihood std.
        self.prior_m = torch.tensor(.9)
        self.prior_s = torch.tensor(1.)
        self.likelihood_s = torch.tensor(5.5)
        self.likelihood_log_pai = torch.tensor([0.])

    def generate_rand(self):
        return np.random.normal(size=(self.n_latent, 1))

    def reparam(self, eps):
        eps = torch.FloatTensor(eps)
        return eps.mul(self.softplus(self.qs)).add(self.qm)

    def compute_elbo(self, x, t):
        z = gmm_sample(self.qm, self.softplus(self.qs), torch.tensor([0.]), 100)

        q_dist = torch.mean(log_gmm(z, [self.qm], [self.softplus(self.qs)], torch.tensor([0.])))
        prior = torch.mean(log_gmm(z, [self.prior_m], [self.prior_s], torch.tensor([0.])))
        likelihood = torch.mean(torch.sum(log_gaussian(t,z.T * x, self.likelihood_s), 0))
        likelihood2 = torch.mean(torch.sum(log_gmm(t, [z.T * x ], [self.likelihood_s], self.likelihood_log_pai), 1))

        kld_mc = q_dist - prior
        loss = likelihood2 - kld_mc
        return loss


if __name__ == '__main__':

    c = MCElbo()
    optimizer = torch.optim.Adam(c.parameters(),lr=0.2)
    X, T = create_unimodal_data(200, w=3.2, scale=5.5)
    x = torch.tensor(X)
    t = torch.tensor(T)

    for i in range(2001):
        loss = -c.compute_elbo(x, t)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        if i % 250 ==0:
            print(c.qm.data.numpy(), (c.softplus(c.qs).data**2).numpy())

    # analytical_posterior_var = ((1/5.5**2)*X.T@ X +1)**-1
    # analytical_posterior_mean = analytical_posterior_var*(0.9+((1/5.5**2)*X.T @ T))
    #
    # wn = np.arange(3.1, 3.5, 0.0001)
    # true_dist = norm(loc=analytical_posterior_mean,
    #                  scale=(analytical_posterior_var)**0.5)
    # q_dist = norm(loc=c.qm.data.numpy(), scale=c.softplus(c.qs).data.numpy())
    # yn = true_dist.pdf(wn).ravel()
    # plt.plot(wn, yn, linewidth=3, label="True Posterior Analytic")
    # plt.plot(wn, q_dist.pdf(wn).ravel(), '--', linewidth=3,
    #          label="Approximation")
    #
    # log_prior = torch.mean(log_gmm(torch.tensor(wn), [c.prior_m], [c.prior_s], torch.tensor([0.])))
    # log_likelihood_2 = torch.sum(log_gmm(t, [torch.tensor(wn) * x], [c.likelihood_s], c.likelihood_log_pai), 1)
    # log_true_posterior = log_likelihood_2 + log_prior
    # log_true_posterior_func = log_true_posterior - torch.max(log_true_posterior)
    # true_posterior = torch.exp(log_true_posterior_func) / torch.sum(torch.exp(log_true_posterior_func))
    # plt.plot(wn, true_posterior, linewidth=3, label="True Posterior")
    #
    # plt.legend()
    # plt.show()

    wn = torch.arange(1.5, 3.5, 0.0001)
    log_likelihood_2 = torch.sum(log_gmm(t, [torch.tensor(wn) * x], [c.likelihood_s], c.likelihood_log_pai), 1)

    log_prior = torch.mean(log_gmm(torch.tensor(wn), [c.prior_m], [c.prior_s], torch.tensor([0.])))

    log_true_posterior = log_likelihood_2 + log_prior

    log_true_posterior_func = log_true_posterior - torch.max(log_true_posterior)

    true_posterior = torch.exp(log_true_posterior_func) / torch.sum(torch.exp(log_true_posterior_func))

    plt.plot(wn, true_posterior, linewidth=3, label="True Posterior")

    # log_true_posterior_func = log_true_posterior - torch.max(log_true_posterior)

    posterior = torch.exp(log_gmm(wn, c.qm, c.softplus(c.qs), torch.tensor([0.])))
    plt.plot(wn, posterior.detach() / torch.sum(posterior.detach()), '--', linewidth=3, label="GMM")

    plt.legend()
    plt.show()