import numpy as np
import utils_2
from svgd import SVGD
import copy
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

n_walkers = 10
walkers = np.array([np.random.rand(n_walkers)]).transpose()
alpha = 5
lr = 0.1
gradient_norm = 1000
n_initial_svgd_steps = 1000
n_update_svgd_steps = 20
alpha_list = [alpha]
energies_list = []
gradients_list = []
grad_difference_list = []
iter = 0
target_distr = utils_2.sho_probability(alpha)
walkers = SVGD().update(walkers, target_distr.dlnprob, n_iter=n_initial_svgd_steps, stepsize=0.01)
while gradient_norm > 1e-4 and iter < 1000:
    print(iter)
    energies_list.append(utils_2.variational_energy_sho(walkers, alpha))
    walkers_grad = walkers[:,0]
    gradient = utils_2.gradient_variational_energy_sho_2(walkers_grad, alpha)
    true_gradient = utils_2.analytical_gradient_sho(alpha)
    alpha = alpha - lr * gradient
    alpha_list.append(alpha)
    gradients_list.append(np.abs(gradient))
    grad_difference_list.append(np.abs(gradient - true_gradient))
    gradient_norm = np.abs(gradient)
    iter += 1
    # Update the particles
    target_distr = utils_2.sho_probability(alpha)
    walkers = SVGD().update(walkers, target_distr.dlnprob, n_iter=n_update_svgd_steps, stepsize=0.01)


print(alpha)
plt.plot(alpha_list, color = 'g', label = 'alpha')
plt.plot(energies_list, color = 'r', label = 'energy')
plt.plot(gradients_list, color = 'b', label = 'gradient norm')
plt.plot(grad_difference_list, color = 'y', label = 'gradient_difference')
plt.xlabel('Iterations')
plt.legend()
plt.show()
