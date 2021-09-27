import numpy as np
import utils_2
from svgd import SVGD
import copy
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

n_repetitions = 20
n_walkers = 500
walkers = np.random.rand(n_walkers)
alpha_values = [0.4, 1, 1.5, 3, 5]
sigma = 0.1
gradient_norm = 1000
lr = 0.1
n_initial_mcmc_steps = 1000
n_update_mcmc_steps = 20
alpha_list = [1]
energies_list = []
grad_list = []
iter = 0
n_iter_array = np.zeros((5, n_repetitions))
#First, we approximate the initial density
for i in range(len(alpha_values)):
    alpha = alpha_values[i]
    print(alpha)
    for j in range(n_repetitions):
        alpha = alpha_values[i]
        print(j)
        walkers = np.random.rand(n_walkers)
        iter = 0
        gradient_norm = 1000
        for k in range(n_initial_mcmc_steps):
            for l in range(n_walkers):
                walkers[l] = utils_2.mcmc_step_sho(walkers[l], sigma, alpha)
        while gradient_norm> 1e-4 and iter < 1000:
            print(iter)
            energies_list.append(utils_2.variational_energy_sho(walkers, alpha))
            gradient = utils_2.gradient_variational_energy_sho(walkers, alpha)
            alpha = alpha - lr*gradient
            alpha_list.append(alpha)
            grad_list.append(np.abs(gradient))
            gradient_norm = np.abs(gradient)
            iter += 1
            #Update the particles
            for k in range(n_update_mcmc_steps):
                for l in range(n_walkers):
                    walkers[l] = utils_2.mcmc_step_sho(walkers[l], sigma, alpha)
        n_iter_array[i][j] = iter
#plt.plot(alpha_list, color = 'g', label = 'alpha')
#plt.plot(energies_list, color = 'r', label = 'energy')
#plt.plot(grad_list, color = 'b', label = 'gradient norm')
#plt.xlabel('Iterations')
#plt.legend()
#plt.show()

mean_iterations = np.mean(n_iter_array, axis = 1)
print(mean_iterations)
std_iterations = np.std(n_iter_array, axis = 1)
print(std_iterations)

np.save('mean_iterations_mcmc_500_sho_samples', mean_iterations)
np.save('std_iterations_mcmc_500_sho_samples', std_iterations)

#print(n_iter_array)

#print(n_iter_list)