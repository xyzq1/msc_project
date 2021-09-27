import numpy as np
import utils_2
from svgd import SVGD
import copy
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import kernel_herding

n_walkers = 1000
n_super_samples = 200
walkers = np.array([np.random.rand(n_walkers)]).transpose()
alpha = 5
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
        walkers[j] = utils_2.mcmc_step_sho(walkers[j], sigma, alpha)
while gradient_norm> 1e-4 and iter < 1000:
    print(iter)
    #Calculate ss before estimating energy
    super_samples = kernel_herding.k_herding(walkers, n_super_samples, gamma)
    energies_list.append(utils_2.variational_energy_sho(super_samples, alpha))
    ss_grad = super_samples[:, 0]
    gradient = utils_2.gradient_variational_energy_sho_2(ss_grad, alpha)
    true_gradient = utils_2.analytical_gradient_sho(alpha)
    alpha = alpha - lr*gradient
    alpha_list.append(alpha)
    grad_list.append(np.abs(gradient))
    grad_difference_list.append(np.abs(gradient - true_gradient))
    gradient_norm = np.abs(gradient)
    iter += 1
    #Update the particles
    for i in range(n_update_mcmc_steps):
        for j in range(n_walkers):
            walkers[j] = utils_2.mcmc_step_sho(walkers[j], sigma, alpha)
plt.plot(alpha_list, color = 'g', label = 'alpha')
plt.plot(energies_list, color = 'r', label = 'energy')
plt.plot(grad_list, color = 'b', label = 'gradient norm')
plt.plot(grad_difference_list, color = 'y', label = 'gradient_difference')
plt.xlabel('Iterations')
plt.legend()
plt.show()

#mean_iterations = np.mean(n_iter_array, axis = 1)
#print(mean_iterations)
#std_iterations = np.std(n_iter_array, axis = 1)
#print(std_iterations)

#np.save('mean_iterations_mcmc_500_sho_samples', mean_iterations)
#np.save('std_iterations_mcmc_500_sho_samples', std_iterations)

#print(n_iter_array)

#print(n_iter_list)