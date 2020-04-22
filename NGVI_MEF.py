import torch
import torch.nn as nn

from distributions import gmm, gaussian, log_gaussian, log_gmm
from utils import nth_derivative
from sampling import gaussian_sample
from plotters import plot_posteriors, generate_animation
from torch.distributions import Categorical

class MCElbo(nn.Module):
    def __init__(self):
        super(MCElbo, self).__init__()
        self.softplus = torch.nn.Softplus()
        self.softmax = torch.nn.Softmax(dim=0)
        self.num_samples = [1]  # Number of samples
        self.n_latent = 5  # Number of latent samples

        self.prior_m = torch.tensor(0.)
        self.prior_s = torch.tensor(10.)
        self.prior_fn = lambda t: log_gaussian(t, self.prior_m, self.prior_s)

        self.q_means = torch.nn.Parameter(torch.tensor([-2., 2.]),
                                          requires_grad=True)
        self.q_stds = torch.nn.Parameter(torch.tensor([.4, .4]),
                                         requires_grad=True)
        self.q_log_pais = torch.nn.Parameter(torch.log(torch.tensor([.5, .5])),
                                             requires_grad=True)
        self.q_fn = lambda t: log_gmm(t, self.q_means,
                                      self.softplus(self.q_stds),
                                      self.q_log_pais)

        self.likelihood_s = torch.tensor([.5, .4])
        self.likelihood_log_pais = torch.log(torch.tensor([.5, .5]))
        self.likelihood_mean = torch.tensor([4.])

        self.likelihood_fn = lambda t: log_gmm(t, self.likelihood_mean,
                                               self.likelihood_s,
                                               self.likelihood_log_pais)

        # self.p = Gaussian(mean=0, sigma= 10) # prior
        # self.q =                GMM(num_components=3, means=[-2.0, 0.0, 4.], variances=[5., 5., 5.], pais=[.33, .33, .33])
        # self.neg_log_ll = NegLogGMM(num_components=3, means=[-4.,2., 5.], variances=[.2, .2, .2], pais=[.3, .5, .2])
        # self.q =                GMM(num_components=2, means=[-2.0,  4.], variances=[5., 5.], pais=[.33, .67])
        # self.neg_log_ll = NegLogGMM(num_components=2, means=[-4., 5.], variances=[.2,  .2], pais=[.8,  .2])

    def compute_elbo(self, z):
        q_likelihood = torch.mean(
            log_gmm(z, self.q_means, self.softplus(self.q_stds),
                    self.q_log_pais))
        prior = torch.mean(self.prior_fn(z))
        log_likelihood = torch.mean(self.likelihood_fn(z))

        elbo = q_likelihood - prior - log_likelihood
        return elbo


def fit_gmm(num_components, beta, num_step=20):
    r = 2
    # num_step = 400
    # num_components = 2
    mcelbo = MCElbo()
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
        c=0
        sampled_z = gaussian_sample(mcelbo.q_means[c], mcelbo.softplus(mcelbo.q_stds[c]),
                            mcelbo.num_samples[c])

        # mixture_comps = Categorical(mcelbo.q_log_pais)
        # component = mixture_comps.sample()
        # sampled_z = objective.q.means[component]
        # sampled_z = mcelbo.q.means[c] + torch.randn(1) * torch.sqrt(mcelbo.q.variances[c])
        # sampled_z.detach()
        all_sampled_z.append(sampled_z)
        dx = nth_derivative(mcelbo.compute_elbo(sampled_z), sampled_z, 1)
        d2xx = nth_derivative(mcelbo.compute_elbo(sampled_z), sampled_z, 2)

        for c in range(num_components):
            delta = gaussian(sampled_z, mcelbo.q_means[c], mcelbo.q_stds[c]) / mcelbo.q_fn(sampled_z)
            all_deltas[c] = delta
            # update mean and variances and ratios
            mcelbo.q_stds[c] = 1 / (1 / mcelbo.q_stds[c] + beta * delta * d2xx)
            mcelbo.q_means[c] = mcelbo.q_means[c] - beta * delta * mcelbo.q_stds[c] * dx

            #objective.q.variances[c] = torch.max(objective.q.variances[c], torch.FloatTensor([1e-6]))

            if mcelbo.q_stds[c] < 0:
                mcelbo.q.variances[c] = 1.0
                print('variance reset!!!!!')


        if i % 10 == 0:
            print("STEP {} ".format(i))
            print("rhos:          {}".format(rho.data.numpy()))
            print("q_pais:        {}".format(torch.exp(mcelbo.q_log_pais).data.numpy()))
            print("q.means:       {}".format(mcelbo.q_means))
            print("q.variances:      {}".format(mcelbo.q_stds))
            print("*" * 30)

            beta = beta * 0.95

        for c in range(num_components):
            # Update ratios
            #rho[c] = rho[c] - beta * 0.01 * (all_deltas[c] - all_deltas[-1]) * objective(sampled_z)
            weight_decay = 0.2
            #rho[c] = (1 - weight_decay*beta)*rho[c] - beta * (all_deltas[c] - all_deltas[-1]) * objective(sampled_z)

            rho[c] = (1) * rho[c] - beta * (all_deltas[c] - all_deltas[-1]) * mcelbo.compute_elbo(sampled_z)

        rho = rho - torch.max(rho)
        mcelbo.q_log_pais.data = torch.softmax(rho, dim=0)

        grads[c] = dx
        hess[c] = d2xx

        x_axis = torch.linspace(-10, 10, 500)
        posterior = gmm(x_axis, mcelbo.q_means, mcelbo.q_stds, mcelbo.q_log_pais)
        normalized_posterior = posterior / torch.sum(posterior)
        true_posterior = torch.exp(-(mcelbo.likelihood_fn(x_axis))) * mcelbo.prior_fn(x_axis)
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
