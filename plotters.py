import matplotlib.pyplot as plt

from matplotlib import animation, rc

import torch


def plot_posteriors(all_normalized_posteriors, normalized_true_posterior, x_axis):
    fig, ax = plt.subplots(figsize=(10, 6))
    # x_axis = torch.linspace(-6, 6, 100)

    ax.plot(x_axis.detach(), all_normalized_posteriors[-1].detach(), '-r', lw=1, label='GMM posterior')
    ax.plot(x_axis.detach(), normalized_true_posterior.detach(), '-g', label='True posterior')
    plt.show()
    plt.savefig('./figs/q_is_2gmm.pdf')


def init_wrapper(approximate_posterior_plt):
    def init():
        approximate_posterior_plt.set_data([], [])
        # approximate_posterior_plt.set_offsets([])
        return (approximate_posterior_plt,)
    return init


def animate_wrapper(approximate_posterior_plt, x_axis, all_normalized_posteriors):
    def animate(i):
        approximate_posterior_plt.set_data(x_axis, all_normalized_posteriors[i])
        return (approximate_posterior_plt,)
    return animate


def generate_animation(all_normalized_posteriors, normalized_true_posterior, x_axis, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    # ax.set_xlim(-4, 4)
    # ax.set_ylim(-.5, 4)

    approximate_posterior_plt, = ax.plot([], [], lw=2, label='GMM posterior')
    ax.plot(x_axis.detach(), normalized_true_posterior.detach(), lw=1, label='True posterior')
    ani = animate_wrapper(approximate_posterior_plt, x_axis, all_normalized_posteriors)
    init = init_wrapper(approximate_posterior_plt)
    anim = animation.FuncAnimation(fig, ani, init_func=init,
                                   frames=len(all_normalized_posteriors), interval=20,
                                   blit=True)
    rc('animation', html='jshtml')
    anim.save(path, dpi=300)
    return anim