import torch
from distributions import log_gmm
from sampling import gmm_sample, gaussian_sample
import matplotlib.pyplot as plt

num_components = 2
class MCElbo(torch.nn.Module):
    def __init__(self):
        super(MCElbo, self).__init__()
        self.softplus = torch.nn.Softplus()
        self.softmax = torch.nn.Softmax(dim=0)
        self.num_samples = [100, 100]  # Number of samples

        self.q_means = torch.nn.Parameter(torch.tensor([-2., 2.]), requires_grad=True)
        self.q_stds = torch.nn.Parameter(torch.tensor([.8, .8]), requires_grad=True)
        self.q_log_pais = torch.nn.Parameter(torch.log(torch.tensor([.5, .5])), requires_grad=True)
        self.q_fn = lambda t: log_gmm(t, self.q_means, self.softplus(self.q_stds), self.q_log_pais)

        self.p_means = torch.tensor([0.])
        self.p_stds = torch.tensor([10.])
        self.p_log_pais = torch.log(torch.tensor([1.]))
        self.p_fn = lambda t: log_gmm(t, self.p_means, self.softplus(self.p_stds), self.p_log_pais)

        self.likelihood_mean = torch.tensor([-2., 2.])
        self.likelihood_s = torch.tensor([-.8, .8])
        self.likelihood_log_pais = torch.log(torch.tensor([.5, .5]))

        self.likelihood_fn = lambda t: log_gmm(t, self.likelihood_mean, self.likelihood_s, self.likelihood_log_pais)

    def compute_loss(self):
        elbo = 0
        # for c in range(num_components):
        c = torch.randint(0,2,[1])
        # print(c)
        z = gaussian_sample(self.q_means[c], self.softplus(self.q_stds[c]), self.num_samples[c])
        q_likelihood = torch.mean(self.q_fn(z))
        prior = torch.mean(self.p_fn(z))
        likelihood = torch.mean(self.likelihood_fn(z))
        # elbo_c = q_likelihood - prior
        elbo_c = q_likelihood - prior - likelihood
        elbo += elbo_c * self.softmax(self.q_log_pais)[c]

        # z = gmm_sample(self.q_means, self.q_stds, self.q_log_pais, self.num_samples)
        # q_likelihood = torch.mean(self.q_fn(z))
        # prior = torch.mean(self.p_fn(z))
        # loss = q_likelihood - prior
        return elbo


if __name__ == '__main__':
    c = MCElbo()
    optimizer = torch.optim.Adam([
        {'params': [c.q_log_pais, c.q_means, c.q_stds], 'lr': 0.0002}])

    for i in range(8000):
        loss = c.compute_loss()
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        if i % 250 == 0:
            print(loss)
            print(i, c.q_means.data.numpy(), c.q_stds.data.numpy(), c.softmax(c.q_log_pais).data.numpy())
            print(i, c.p_means.data.numpy(), c.p_stds.data.numpy(), c.softmax(c.p_log_pais).data.numpy())

    wn = torch.arange(-4., 5.5, 0.0001)
    log_likelihood = c.likelihood_fn(wn)

    log_prior = c.p_fn(wn)

    log_true_posterior = log_likelihood + log_prior
    log_true_posterior = log_true_posterior - torch.max(log_true_posterior)
    true_posterior = torch.exp(log_true_posterior) / torch.sum(torch.exp(log_true_posterior))

    plt.plot(wn, true_posterior, linewidth=3, label="True Posterior")

    posterior = torch.exp(c.q_fn(wn))
    plt.plot(wn, posterior.detach() / torch.sum(posterior.detach()), '--', linewidth=3, label="GMM")

    plt.legend()
    plt.show()