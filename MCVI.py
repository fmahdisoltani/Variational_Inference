import torch
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from distributions import log_gaussian, log_gmm
from sampling import gmm_sample, log_normalize
from dataset import create_bimodal_data_fix


class MCElbo(torch.nn.Module):
    def __init__(self):
        super(MCElbo, self).__init__()
        self.num_samples = 100  # Number of latent samples
        self.softplus = torch.nn.Softplus()
        # self.q_means = torch.nn.Parameter(torch.randn(2) * 1, requires_grad=True)
        self.q_means = torch.nn.Parameter(torch.tensor([-2., 2.]), requires_grad=True)
        self.q_stds = torch.nn.Parameter(torch.tensor([.4, .4]), requires_grad=True)
        self.q_log_pais = torch.nn.Parameter(torch.log(torch.tensor([.5, .5])), requires_grad=True)

        self.prior_m = torch.tensor(0.9)
        self.prior_s = torch.tensor(10.)
        self.likelihood_s = torch.tensor([.5, .4])
        self.likelihood_log_pais= torch.log(torch.tensor([.5, .5]))
        self.likelihood_fn = lambda t, mean: log_gmm(t, mean, self.likelihood_s, self.likelihood_log_pais)

        self.prior_fn = lambda t: log_gaussian(t, self.prior_m, self.prior_s)

    def compute_elbo(self, x, t):
        num_components = len(self.q_means)
        loss = 0

        for c in range(num_components):
            z = gmm_sample([self.q_means[c:c+1]], [self.softplus(self.q_stds[c:c+1])], torch.log(torch.tensor([1.])), self.num_samples)
            q_likelihood = torch.mean(log_gmm(z, self.q_means, self.softplus(self.q_stds), self.q_log_pais))
            prior = torch.mean(self.prior_fn(z))
            likelihood = torch.mean(self.likelihood_fn(t, [z.T * x, (z.T + bias) * x]))
            loss_c = likelihood - q_likelihood + prior
            loss += loss_c * torch.exp(self.q_log_pais[c])


        # z = gmm_sample(self.q_means, self.softplus(self.q_stds), self.q_log_pais, self.num_samples)
        # q_likelihood = torch.mean(log_gmm(z, self.q_means, self.softplus(self.q_stds), self.q_log_pais))
        # prior = torch.mean(self.prior_fn(z))
        # likelihood = torch.mean(torch.sum(self.likelihood_fn(t, [z.T * x, (z.T + bias) * x]), 1))
        # loss = likelihood - q_likelihood + prior
        return loss


if __name__ == '__main__':
    torch.manual_seed(10.)
    bias = 2.

    X, T = create_bimodal_data_fix(bias, num_samples=200)
    x = torch.Tensor(X)
    t = torch.Tensor(T)

    c = MCElbo()
    optimizer = torch.optim.Adam([
        {'params': c.q_log_pais, 'lr': 0.002},
        {'params': c.q_means, 'lr': 0.5},
        {'params': c.q_stds, 'lr': 0.5},
    ])
    all_normalized_posterios = []
    for i in range(30):
        loss = -c.compute_elbo(x, t)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        # if i % 10 == 0:
        #     print(optimizer.param_groups[0])
        if i % 250 == 0:
            print(i, c.q_means.data.numpy(), (c.softplus(c.q_stds).data ** 2).numpy(), torch.exp(log_normalize(c.q_log_pais.data)))

    wn = torch.arange(-3, 5.5, 0.0001)
    log_likelihood = torch.sum(c.likelihood_fn(t, [wn * x, (wn+bias)*x]), 1)

    log_prior = c.prior_fn(wn)

    log_true_posterior = log_likelihood + log_prior

    log_true_posterior_func = log_true_posterior - torch.max(log_true_posterior)

    true_posterior = torch.exp(log_true_posterior_func) / torch.sum(torch.exp(log_true_posterior_func))

    plt.plot(wn, true_posterior, linewidth=3, label="True Posterior")

    posterior = torch.exp(log_gmm(wn, c.q_means, c.softplus(c.q_stds), c.q_log_pais))
    plt.plot(wn, posterior.detach()/torch.sum(posterior.detach()), '--', linewidth=3, label="GMM")

    plt.legend()
    plt.show()