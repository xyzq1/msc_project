import numpy as np
import utils_2
from svgd import SVGD
import copy
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

n_walkers = 100
walkers = np.random.rand(n_walkers, 6)
alpha = 0.1
sigma = 0.1
gradient_norm = 1000
lr = 0.5
n_initial_mcmc_steps = 1000
n_update_mcmc_steps = 20
alpha_list = [alpha]
energies_list = []
grad_list = []
iter = 0
#First, we approximate the initial density
for i in range(n_initial_mcmc_steps):
    print(i)
    for j in range(n_walkers):
        walkers[j] = utils_2.mcmc_step_he(walkers[j], sigma, alpha)
while gradient_norm> 1e-4 and iter < 100:
    print(iter)
    energies_list.append(utils_2.variational_energy_he(walkers, alpha))
    gradient = utils_2.gradient_variational_energy_he(walkers, alpha)
    alpha = alpha - lr*gradient
    alpha_list.append(alpha)
    grad_list.append(np.abs(gradient))
    gradient_norm = np.abs(gradient)
    iter += 1
    #Update the particles
    for i in range(n_update_mcmc_steps):
        for j in range(n_walkers):
            walkers[j] = utils_2.mcmc_step_he(walkers[j], sigma, alpha)

print(energies_list[-1])
plt.plot(alpha_list, color = 'g', label = 'alpha')
plt.plot(energies_list, color = 'r', label = 'energy')
plt.plot(grad_list, color = 'b', label = 'gradient norm')
plt.xlabel('Iterations')
plt.legend()
plt.show()
