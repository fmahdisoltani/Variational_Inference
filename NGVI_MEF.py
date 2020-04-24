import torch
import torch.nn as nn

from distributions import Gaussian, GMM, NegLogGMM, gmm, gaussian
from utils import nth_derivative
from plotters import plot_posteriors, generate_animation
from torch.distributions import Categorical
from distributions import log_gaussian, log_gmm
from sampling import gmm_sample


class H(nn.Module):
    def __init__(self):
        super().__init__()
        # self.p = Gaussian(mean=0, sigma= 10) # prior
        # self.q =                GMM(num_components=2, means=[-2., 2.], variances=[.5, .5], pais=[.3, .7])
        # self.neg_log_ll = NegLogGMM(num_components=2, means=[-4., 4.], variances=[.2, .2], pais=[.7, .3])


        self.softplus = torch.nn.Softplus()
        self.softmax = torch.nn.Softmax(dim=0)

        self.prior_m = torch.tensor(0.)
        self.prior_s = torch.tensor(10.)
        self.prior_fn = lambda t: log_gaussian(t, self.prior_m, self.prior_s)

        self.likelihood_s = torch.tensor([.2, .2])
        self.likelihood_log_pais = torch.log(torch.tensor([.3, .7]))
        self.likelihood_mean = torch.tensor([-4., 4.])

        self.likelihood_fn = lambda t: log_gmm(t, self.likelihood_mean,
                                               self.likelihood_s,
                                               self.likelihood_log_pais)

        self.q_means = torch.tensor([-2., 2.])
        self.q_stds = torch.tensor([.4, .4])
        self.q_log_pais = torch.log(torch.tensor([.5, .5]))
        self.q_fn = lambda t: log_gmm(t, self.q_means,
                                      self.q_stds,
                                      self.q_log_pais)

    def forward(self, x):
        h = self.q_fn(x) - self.prior_fn(x) - self.likelihood_fn(x)
        return h


def fit_gmm(num_components, beta, num_step=20, num_samples=1):
    objective = H()

    all_deltas = torch.zeros(num_components)
    grads = torch.zeros(num_components)
    hess = torch.zeros(num_components)
    all_sampled_z = torch.zeros(num_components)
    rho = torch.zeros(num_components)

    all_posteriors = []
    all_normalized_posteriors = []

    for i in range(num_step):

        # mixture_comps = Categorical(torch.exp(objective.q_log_pais))
        # component = mixture_comps.sample()
        # sampled_z = objective.q_means[component] + torch.randn(1) * objective.q_stds[component]

        sampled_z = gmm_sample(objective.q_means, objective.q_stds,
                               objective.q_log_pais, num_samples)
        sampled_z = torch.nn.Parameter(sampled_z)

        dx = nth_derivative(objective(sampled_z), sampled_z, 1)
        d2xx = nth_derivative(objective(sampled_z), sampled_z, 2)
        for c in range(num_components):
            delta = gaussian(sampled_z, objective.q_means[c],
                             objective.q_stds[c]) / torch.exp(
                objective.q_fn(sampled_z))
            all_deltas[c] = delta

            # update mean and variances and ratios
            q_var = objective.q_stds[c] ** 2
            q_var = 1 / (1 / q_var + beta * delta * d2xx)
            objective.q_stds[c] = torch.sqrt(q_var)
            objective.q_means[c] = objective.q_means[
                                       c] - beta * delta * q_var * dx
            # objective.q_stds[c] = torch.sqrt(torch.max(q_var, torch.FloatTensor([1e-6])))

        if i % 10 == 0:
            print("STEP {} ".format(i))
            print("rhos:          {}".format(rho.data.numpy()))
            print("q_pais:        {}".format(
                torch.exp(objective.q_log_pais).data.numpy()))
            print("q.means:       {}".format(objective.q_means))
            print("q.stds:      {}".format(objective.q_stds))
            print("*" * 30)
            beta = beta * .95

        for c in range(num_components):
            # Update ratios
            rho[c] = objective.q_log_pais[c] - objective.q_log_pais[-1]
            rho[c] = rho[c] - beta * 0.1 * (
                        all_deltas[c] - all_deltas[-1]) * objective(sampled_z)

        # rho = rho - torch.max(rho)
        objective.q_log_pais = rho - torch.logsumexp(rho, axis=0,
                                                     keepdims=False)
        # torch.log(torch.softmax(rho, dim=0))

        # grads[c] = dx
        # hess[c] = d2xx

        x_axis = torch.arange(-5, 5, 0.0001)
        posterior = gmm(x_axis, objective.q_means, objective.q_stds,
                        torch.exp(objective.q_log_pais))
        normalized_posterior = posterior / torch.sum(posterior)
        log_true_posterior = objective.likelihood_fn(
            x_axis) + objective.prior_fn(x_axis)

        true_posterior = torch.exp(log_true_posterior)
        normalized_true_posterior = true_posterior / torch.sum(true_posterior)

        #     ax1 = fig.add_subplot(121)
        #     plt.plot(x_axis.detach(), normalized_posterior.detach(), '-b')
        #     plt.plot(x_axis.detach(), true_posterior, '-r')
        #     plt.show()

        all_normalized_posteriors.append(normalized_posterior.detach())
        all_posteriors.append(posterior.detach())

    return all_normalized_posteriors, normalized_true_posterior, x_axis


if __name__ == '__main__':
    all_normalized_posteriors, normalized_true_posterior, x_axis = fit_gmm(
        num_step=501, num_components=2, beta=0.1, num_samples=1)
    plot_posteriors(all_normalized_posteriors, normalized_true_posterior,
                    x_axis)

    generate_animation(all_normalized_posteriors, normalized_true_posterior,
                       x_axis, path='./figs/gmm_3.mp4')