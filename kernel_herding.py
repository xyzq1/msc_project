from sklearn.gaussian_process.kernels import RBF
from scipy.optimize import minimize
import numpy as np

def cost_function_kh(x, walkers, super_samples, gamma):
    #Calculates the expectation kernel term from the KH loss
    kernel = RBF(gamma)
    n_walkers = walkers.shape[0]
    n_super_samples = super_samples.shape[0]
    return np.sum(kernel(x, walkers))/n_walkers - np.sum(kernel(x, super_samples))/(n_super_samples + 1)


def k_herding(walkers, n_super_samples, gamma):
    dim_samples = walkers.shape[1]
    n_walkers = walkers.shape[0]
    super_samples = np.zeros((n_super_samples, dim_samples))
    initializiation = np.random.rand(dim_samples)
    i = 0
    while i < n_super_samples:
        f = lambda x: -cost_function_kh(x, walkers, super_samples, gamma)
        results = minimize(f,
                           initializiation,
                           method='nelder-mead',
                           options={'xtol': 1e-9, 'disp': False})

        if np.min(results.x) < np.min(walkers) or np.max(results.x) > np.max(walkers):
            initializiation = walkers[np.random.choice(n_walkers)]
            continue

        super_samples[i,:] = results.x

        #We choose the new initialization point to be the best supersample
        ss_costs = np.array([])
        for j in range(i + 1):
            ss_costs= np.append(ss_costs,-cost_function_kh(super_samples[j,:], walkers, super_samples, gamma))
        best_ss = np.argmin(ss_costs)
        initializiation = super_samples[best_ss, :]
        i = i + 1
    return super_samples


