import numpy as np
import utils_2
from svgd import SVGD
import copy
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import kernel_herding

n_walkers = 1000
n_super_samples = 100
walkers = np.random.rand(n_walkers, 3)
alpha = 0.5
sigma = 0.1
gamma = 0.1
gradient_norm = 1000
lr = 0.1
n_initial_mcmc_steps = 1000
n_update_mcmc_steps = 20
alpha_list = [alpha]
energies_list = []
grad_list = []
grad_difference_list = []
iter = 0
#First, we approximate the initial density
for i in range(n_initial_mcmc_steps):
    for j in range(n_walkers):
        walkers[j] = utils_2.mcmc_step_h(walkers[j], sigma, alpha)
while gradient_norm> 1e-4 and iter < 100:
    print(iter)
    #Calculate ss before estimating energy
    super_samples = kernel_herding.k_herding(walkers, n_super_samples, gamma)
    energies_list.append(utils_2.variational_energy_h(super_samples, alpha))
    gradient = utils_2.gradient_variational_energy_h_2(super_samples, alpha)
    true_gradient = utils_2.analytical_gradient_h(alpha)
    alpha = alpha - lr*gradient
    alpha_list.append(alpha)
    grad_list.append(np.abs(gradient))
    grad_difference_list.append(np.abs(gradient - true_gradient))
    gradient_norm = np.abs(gradient)
    iter += 1
    #Update the particles
    for i in range(n_update_mcmc_steps):
        for j in range(n_walkers):
            walkers[j] = utils_2.mcmc_step_h(walkers[j], sigma, alpha)
plt.plot(alpha_list, color = 'g', label = 'alpha')
plt.plot(energies_list, color = 'r', label = 'energy')
plt.plot(grad_list, color = 'b', label = 'gradient norm')
plt.plot(grad_difference_list, color = 'y', label = 'gradient_difference')
plt.xlabel('Iterations')
plt.legend()
plt.show()
