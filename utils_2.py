import numpy as np
import copy
import math
from scipy.optimize import minimize
from scipy import special

def local_energy_sho(alpha, x):
    local_energy = alpha + (0.5 - 2*alpha**2)*x**2
    return local_energy

def variational_energy_sho(walkers, alpha):
    local_energies = [local_energy_sho(alpha, walker) for walker in walkers]
    energy = np.mean(local_energies)
    return energy

def analytical_energy_sho(alpha):
    return (4*alpha**2 + 1)/(8*alpha)

def mcmc_proposal_sho(walker, sigma):
    proposal = np.random.normal(walker, sigma)
    return proposal

def mcmc_step_sho(walker, sigma, alpha):
    proposed_walker = mcmc_proposal_sho(walker, sigma)
    log_prob_ratio = 2*alpha*(walker**2 - proposed_walker**2)
    prob_ratio = np.exp(log_prob_ratio)
    coin = np.random.rand()
    if coin <= prob_ratio:
        new_walker = proposed_walker
    else:
        new_walker = walker
    return new_walker

def gradient_variational_energy_sho(walkers, alpha):
    local_energies = [local_energy_sho(alpha, walker) for walker in walkers]
    n_walkers = walkers.shape[0]
    grad_log_psi = -walkers**2 + 1/(4*alpha)
    gradient = 2 * np.dot(local_energies - np.mean(local_energies), grad_log_psi) / n_walkers
    return gradient

def gradient_variational_energy_sho_2(walkers, alpha):
    local_energies = [local_energy_sho(alpha, walker) for walker in walkers]
    n_walkers = walkers.shape[0]
    grad_log_psi = -walkers**2
    gradient = 2 * np.dot(local_energies - np.mean(local_energies), grad_log_psi) / n_walkers
    return gradient

class sho_probability:

    def __init__(self, alpha):
        self.alpha = alpha

    def dlnprob(self, walkers):
        lngrad = -4*self.alpha*walkers
        return lngrad

def analytical_gradient_sho(alpha):
    true_grad = 2*(0.125 - 0.09375/alpha**2) + (1 + 4*alpha**2)/(16*alpha**2)
    return true_grad

def local_energy_h(walker, alpha):
    r_walker = np.linalg.norm(walker)
    e_local = -0.5*(alpha**2 - 2*alpha/r_walker) - 1/r_walker
    return e_local

def analytical_energy_h(alpha):
    true_energy = 3.5 * alpha * (-1.14286 + alpha)
    return true_energy

def variational_energy_h(walkers, alpha):
    local_energies = [local_energy_h(walker, alpha) for walker in walkers]
    return np.mean(local_energies)

def mcmc_proposal_h(walker, sigma):
    proposal = np.random.multivariate_normal(walker, sigma*np.eye(3))
    return proposal

def mcmc_step_h(walker, sigma, alpha):
    walker_proposal = mcmc_proposal_h(walker, sigma)
    r_norm_prop = np.linalg.norm(walker_proposal)
    r_walker = np.linalg.norm(walker)
    log_prob_ratio = 2*alpha*(r_walker - r_norm_prop)
    prob_ratio = np.exp(log_prob_ratio)
    coin = np.random.rand()
    if coin <= prob_ratio:
        new_walker = walker_proposal
    else:
        new_walker = walker
    return new_walker

def gradient_variational_energy_h(walkers, alpha):
    local_energies = [local_energy_h(walker, alpha) for walker in walkers]
    r_walkers = np.linalg.norm(walkers, axis = 1)
    n_walkers = walkers.shape[0]
    grad_log_psi = -r_walkers**2 + 3/(4*alpha)
    gradient = 2 * np.dot(local_energies - np.mean(local_energies), grad_log_psi) / n_walkers
    return gradient

def gradient_variational_energy_h_2(walkers, alpha):
    local_energies = [local_energy_h(walker, alpha) for walker in walkers]
    r_walkers = np.linalg.norm(walkers, axis = 1)
    n_walkers = walkers.shape[0]
    grad_log_psi = -r_walkers**2
    gradient = 2 * np.dot(local_energies - np.mean(local_energies), grad_log_psi) / n_walkers
    return gradient

class hydrogen_probability:

    def __init__(self, alpha):
        self.alpha = alpha

    def dlnprob(self, walkers):
        lngrad = -4*self.alpha*walkers
        return lngrad

def analytical_gradient_sho(alpha):
    true_grad = 2*(0.125 - 0.09375/alpha**2) + (1 + 4*alpha**2)/(16*alpha**2)
    return true_grad

def analytical_gradient_h(alpha):
    true_grad = 2*(-0.797885*np.sqrt(alpha) + 0.797885/np.sqrt(alpha) + 0.375*alpha) + 3*(-1.59577 + 1.59577*alpha - 0.5*np.power(alpha, 1.5))/(2*np.sqrt(alpha))
    return true_grad

def prob_density_he(walker, alpha):
    r_1 = walker[0:3]
    r_2 = walker[3:6]
    r1_norm = np.linalg.norm(r_1)
    r2_norm = np.linalg.norm(r_2)
    r_12 = np.linalg.norm(r_1 - r_2)
    wavefunction = np.exp(-2*r1_norm)*np.exp(-2*r2_norm)*np.exp(r_12/(2*(1+alpha*r_12)))
    prob = wavefunction**2
    return prob

def local_energy_he(walker, alpha):
    r_1 = walker[0:3]
    r_2 = walker[3:6]
    r1_norm = np.linalg.norm(r_1)
    r2_norm = np.linalg.norm(r_2)
    r_12 = np.linalg.norm(r_1 - r_2)
    r_1_unit = r_1/r1_norm
    r2_unit = r_2/r2_norm
    local_e = -4 + np.dot(r_1_unit-r2_unit,r_1-r_2)/(r_12*(1 + alpha*r_12)**2) - 1/(r_12*(1 + alpha*r_12)**3) - 1/(4*(1 + alpha*r_12)**4) + 1/r_12
    return local_e

def variational_energy_he(walkers, alpha):
    local_energies = [local_energy_he(walker, alpha) for walker in walkers]
    energy = np.mean(local_energies)
    return energy

def log_prob_he(walker, alpha):
    r_1 = walker[0:3]
    r_2 = walker[3:6]
    r1_norm = np.linalg.norm(r_1)
    r2_norm = np.linalg.norm(r_2)
    r_12 = np.linalg.norm(r_1 - r_2)
    log_prob = -2*r1_norm - 2*r2_norm + r_12/(2*(1 + alpha*r_12))
    log_prob = 2*log_prob
    return log_prob

def mcmc_proposal_he(walker, sigma):
    new_walker = np.random.multivariate_normal(walker, sigma*np.eye(6))
    return new_walker

def mcmc_step_he(walker, sigma, alpha):
    prop_walker = mcmc_proposal_he(walker, sigma)
    log_prob_prop = log_prob_he(prop_walker, alpha)
    log_prob_init = log_prob_he(walker, alpha)
    coin = np.random.rand()
    if coin <= np.exp(log_prob_prop - log_prob_init):
        new_walker = prop_walker
    else:
        new_walker = walker
    return new_walker


def gradient_variational_energy_he(walkers, alpha):
    local_energies = [local_energy_he(walker, alpha) for walker in walkers]
    r_1 = walkers[:, 0:3]
    r_2 = walkers[:, 3:6]
    r_12 = r_1 - r_2
    r_12 = np.linalg.norm(r_12, axis = 1)
    n_walkers = walkers.shape[0]
    grad_log_psi = -0.5*r_12**2/(1 + alpha*r_12)**2
    gradient = 2 * np.dot(local_energies - np.mean(local_energies), grad_log_psi) / n_walkers
    return gradient

class he_probability:

    def __init__(self, alpha):
        self.alpha = alpha

    def dlnprob(self, walkers):
        r_1 = walkers[:, 0:3]
        r_1_norm = np.linalg.norm(r_1, axis = 1)
        r_1_norm = np.array([r_1_norm]).transpose()
        r_2 = walkers[:, 3:6]
        r_2_norm = np.linalg.norm(r_2, axis = 1)
        r_2_norm = np.array([r_2_norm]).transpose()
        r_12 = r_1 - r_2
        r_12 = np.linalg.norm(r_12, axis=1)
        r_12 = np.array([r_12]).transpose()
        r_1_grad = -4*r_1/r_1_norm + (r_2 - r_1)*self.alpha/(1 + r_12*self.alpha)**2 + (r_1 - r_2)/(r_12*(1 + self.alpha*r_12))
        r_2_grad = -4*r_2/r_2_norm + (-r_2 + r_1)*self.alpha/(1 + r_12*self.alpha)**2 + (r_2 - r_1)/(r_12*(1 + self.alpha*r_12))
        lngrad = np.concatenate((r_1_grad, r_2_grad), axis = 1)
        return lngrad




