import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm
from dataset import create_unimodal_data
from distributions import log_gaussian, log_gmm
from sampling import gmm_sample, gaussian_sample

plt.style.use('ggplot')


class MCElbo(torch.nn.Module):
    def __init__(self):
        super(MCElbo, self).__init__()
        self.softplus = torch.nn.Softplus()
        self.softmax = torch.nn.Softmax(dim=0)
        self.num_samples = [5]  # Number of samples
        self.n_latent = 5  # Number of latent samples

        # The parameters we adjust during training.
        self.q_means = torch.nn.Parameter(torch.tensor([2.]), requires_grad=True)
        self.q_stds = torch.nn.Parameter(torch.tensor([.8]), requires_grad=True)
        self.q_log_pais = self.q_log_pais = torch.nn.Parameter(torch.log(torch.tensor([1.])), requires_grad=True)
#torch.tensor([0.])
        self.q_fn = lambda t: log_gmm(t, self.q_means, self.softplus(self.q_stds), self.q_log_pais)


        # create holders for prior mean and std, and likelihood std.
        self.p_means = torch.tensor([-4.])
        self.p_stds = torch.tensor([1.])
        self.p_log_pais = torch.log(torch.tensor([1.]))
        self.p_fn = lambda t: log_gmm(t, self.p_means, self.softplus(self.p_stds), self.p_log_pais)


        # self.likelihood_mean = torch.tensor([4.])
        self.likelihood_s = torch.tensor([5.5])
        self.likelihood_log_pais = torch.log(torch.tensor([1.]))

        # self.likelihood_fn = lambda t: log_gmm(t, self.likelihood_mean, self.likelihood_s, self.likelihood_log_pais)



    def compute_elbo(self, x, t):
        elbo = 0
        for c in range(len(self.q_means)):
            z = gaussian_sample(self.q_means[c], self.softplus(self.q_stds[c]), self.num_samples[c])

            q_likelihood = torch.mean(self.q_fn(z))
            prior = torch.mean(self.p_fn(z))
            # likelihood = torch.mean(torch.sum(log_gaussian(t,z.T * x, self.likelihood_s), 0))
            # likelihood2 = torch.mean(torch.sum(log_gmm(t, [z.T * x ], [self.likelihood_s], self.likelihood_log_pai), 1))
            # log_likelihood1 = torch.mean(self.likelihood_fn(z))
            log_likelihood2 = torch.mean(torch.sum(log_gmm(t, [z.T * x], self.likelihood_s,
                        self.likelihood_log_pais), 1))
            # likelihood = torch.mean(torch.sum(log_norm(t, x * z.T,
            #                                            self.likelihood_s), 0))
            log_likelihood = log_likelihood2
            elbo_c = q_likelihood - prior - log_likelihood
            elbo += elbo_c * self.softmax(self.q_log_pais)[c]

        return elbo


if __name__ == '__main__':

    mcelbo = MCElbo()
    optimizer = torch.optim.Adam(mcelbo.parameters(), lr=0.02)
    x, t = create_unimodal_data(200, w=3.2, scale=5.5)

    for i in range(2001):
        loss = mcelbo.compute_elbo(x, t)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        if i % 250 == 0:
            print(mcelbo.q_means.data.numpy(),
                  (mcelbo.softplus(mcelbo.q_stds).data**2).numpy())

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
#################################################################
    # wn = torch.arange(-4., 5.5, 0.0001)
    # # log_likelihood_2 = torch.sum(log_gmm(t, [torch.tensor(wn) * x], [c.likelihood_s], c.likelihood_log_pai), 1)
    # log_likelihood = log_gmm(wn, c.likelihood_mean, c.likelihood_s, c.likelihood_log_pai)
    # log_prior = log_gmm(wn, c.prior_m, c.prior_s, c.p_log_pais)
    #
    # log_true_posterior = log_likelihood + log_prior
    # log_true_posterior_func = log_true_posterior - torch.max(log_true_posterior)
    # true_posterior = torch.exp(log_true_posterior_func) / torch.sum(torch.exp(log_true_posterior_func))
    #
    # plt.plot(wn, true_posterior, linewidth=3, label="True Posterior")
    #
    # # log_true_posterior_func = log_true_posterior - torch.max(log_true_posterior)
    #
    # posterior = torch.exp(log_gmm(wn, c.qm, c.softplus(c.qs), torch.tensor([0.])))
    # plt.plot(wn, posterior.detach() / torch.sum(posterior.detach()), '--', linewidth=3, label="GMM")
    #
    # plt.legend()
    # plt.show()

    wn = torch.arange(1., 3.5 , 0.0001)
    # log_likelihood1 = mcelbo.likelihood_fn(wn)
    log_likelihood2 = torch.sum(log_gmm(t, [torch.tensor(wn) * x],
                                       mcelbo.likelihood_stds, mcelbo.likelihood_log_pais), 1)
    log_likelihood = log_likelihood2
    log_prior = mcelbo.p_fn(wn)

    log_true_posterior = log_likelihood + log_prior
    log_true_posterior = log_true_posterior - torch.max(log_true_posterior)
    true_posterior = torch.exp(log_true_posterior) / torch.sum(
        torch.exp(log_true_posterior))

    plt.plot(wn, true_posterior, linewidth=3, label="True Posterior")

    posterior = torch.exp(mcelbo.q_fn(wn))
    plt.plot(wn, posterior.detach() / torch.sum(posterior.detach()), '--',
             linewidth=3, label="GMM")
    # plt.plot(wn, torch.exp(log_likelihood))
    plt.legend()
    plt.show()