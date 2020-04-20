import torch
import torch.nn as nn

from distributions import Gaussian, GMM, NegLogGMM, gmm, gaussian
from utils import nth_derivative
from plotters import plot_posteriors, generate_animation
from torch.distributions import Categorical


class H(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = Gaussian(mean=0, sigma= 10) # prior
        # self.q =                GMM(num_components=3, means=[-2.0, 0.0, 4.], variances=[5., 5., 5.], pais=[.33, .33, .33])
        # self.neg_log_ll = NegLogGMM(num_components=3, means=[-4.,2., 5.], variances=[.2, .2, .2], pais=[.3, .5, .2])
        self.q =                GMM(num_components=2, means=[-2.0,  4.], variances=[5., 5.], pais=[.33, .67])
        self.neg_log_ll = NegLogGMM(num_components=2, means=[-4., 5.], variances=[.2,  .2], pais=[.8,  .2])

    def forward(self, x):
        h = torch.log(self.q(x)) - torch.log(self.p(x)) + self.neg_log_ll(x)
        return h


def fit_gmm(num_components, beta, num_step=20):
    r = 2
    # num_step = 400
    # num_components = 2
    objective = H()
    # beta = 0.001

    # x = torch.FloatTensor(1).uniform_(-r, r)
    # x.requires_grad = True

    all_deltas = torch.zeros(num_components)
    grads = torch.zeros(num_components)
    hess = torch.zeros(num_components)
    all_sampled_z = []
    rho = torch.zeros(num_components)

    all_posteriors = []
    all_normalized_posteriors = []

    for i in range(num_step):

        mixture_comps = Categorical(objective.q.pais)
        component = mixture_comps.sample()
        # sampled_z = objective.q.means[component]
        sampled_z = objective.q.means[component] + torch.randn(1) * torch.sqrt(objective.q.variances[component])
        sampled_z.detach()
        all_sampled_z.append(sampled_z)
        dx = nth_derivative(objective(sampled_z), sampled_z, 1)
        d2xx = nth_derivative(objective(sampled_z), sampled_z, 2)

        for c in range(num_components):
            delta = gaussian(sampled_z, objective.q.means[c], objective.q.variances[c]) / objective.q(sampled_z)
            all_deltas[c] = delta
            # update mean and variances and ratios
            objective.q.variances[c] = 1 / (1 / objective.q.variances[c] + beta * delta * d2xx)
            objective.q.means[c] = objective.q.means[c] - beta * delta * objective.q.variances[c] * dx

            #objective.q.variances[c] = torch.max(objective.q.variances[c], torch.FloatTensor([1e-6]))

            if objective.q.variances[c] < 0:
                objective.q.variances[c] = 1.0
                print('variance reset!!!!!')


        if i % 10 == 0:
            print("STEP {} ".format(i))
            print("rhos:          {}".format(rho))
            print("q.pais:        {}".format(objective.q.pais))
            print("q.means:       {}".format(objective.q.means))
            print("q.variances:      {}".format(objective.q.variances))
            print("*" * 30)

            beta = beta * 0.95

        for c in range(num_components):
            # Update ratios
            #rho[c] = rho[c] - beta * 0.01 * (all_deltas[c] - all_deltas[-1]) * objective(sampled_z)
            weight_decay = 0.2
            #rho[c] = (1 - weight_decay*beta)*rho[c] - beta * (all_deltas[c] - all_deltas[-1]) * objective(sampled_z)

            rho[c] = (1) * rho[c] - beta * (all_deltas[c] - all_deltas[-1]) * objective(sampled_z)

        rho = rho - torch.max(rho)
        objective.q.pais = torch.softmax(rho, dim=0)

        grads[c] = dx
        hess[c] = d2xx

        x_axis = torch.linspace(-10, 10, 500)
        posterior = gmm(x_axis, objective.q.means, objective.q.variances, objective.q.pais)
        normalized_posterior = posterior / torch.sum(posterior)
        true_posterior = torch.exp(-(objective.neg_log_ll(x_axis))) * objective.p(x_axis)
        normalized_true_posterior = true_posterior / torch.sum(true_posterior)

        #     ax1 = fig.add_subplot(121)
        #     plt.plot(x_axis.detach(), normalized_posterior.detach(), '-b')
        #     plt.plot(x_axis.detach(), true_posterior, '-r')
        #     plt.show()

        all_normalized_posteriors.append(normalized_posterior.detach())
        all_posteriors.append(posterior.detach())

    return all_normalized_posteriors, normalized_true_posterior, x_axis


if __name__ == '__main__':

    all_normalized_posteriors, normalized_true_posterior, x_axis = fit_gmm(num_step=200, num_components=2, beta=0.0001)

    plot_posteriors(all_normalized_posteriors, normalized_true_posterior, x_axis)

    generate_animation(all_normalized_posteriors, normalized_true_posterior, x_axis, path='./figs/gmm_3.mp4')
