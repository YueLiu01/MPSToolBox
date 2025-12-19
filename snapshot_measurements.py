import numpy as np
import tenpy.linalg.np_conserved as npc
import scipy
import matplotlib.pyplot as plt
np.set_printoptions(precision=5, suppress=True, linewidth=100)
plt.rcParams['figure.dpi'] = 150
import tenpy
import tenpy.linalg.np_conserved as npc
from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS
from tenpy.models.tf_ising import TFIChain
from tenpy.networks.site import SpinSite
from tenpy.networks.mps import TransferMatrix
from tenpy.models.model import CouplingMPOModel, NearestNeighborModel
from tenpy.models.lattice import Chain
from tenpy.linalg import np_conserved as npc
import numpy as np
#tenpy.tools.misc.setup_logging(to_stdout="INFO")
from tenpy.models.lattice import Lattice
from tenpy.models.model import CouplingModel,MPOModel
from tenpy.networks.site import SpinHalfSite, kron
from scipy.optimize import curve_fit
import pickle
import MPSToolBox as my


psi = my.load_pkl("../wavefunctions/CritIsingModel_L10_chi300_PBC_.pkl")
beta_anc = 0.2
beta_sys = np.inf
povm_1 = my.weak_measurement_pauli(my.sZ, beta=beta_anc, real=True)
povm_2 = my.weak_measurement_pauli(my.sZ, beta=beta_sys, real=True)
N = 100 # Number of measurements

outcomes_anc = []
outcomes_sys = []
for seed in range(N):
    rng = np.random.default_rng(seed=seed)
    s, weight = my.sample_multi_povm_measurements(psi, first_site=0, ops=[[povm_1, povm_2]]*psi.L, rng=rng)
    s = np.array(s)
    s_anc = s[:,0]
    s_sys = s[:,1]
    outcomes_anc.append(s_anc)
    outcomes_sys.append(s_sys)
    
outcomes_anc = np.array(outcomes_anc)
outcomes_sys = np.array(outcomes_sys)

observable = outcomes_sys[:,0]


def second_moment_snapshot(outcomes_anc, observable):
    r'''
    Consider a system attached to some ancilla degrees of freedom. We are interested in computing \sum_{p_s} p_s \langle O \rangle^2_s for some observable O after measuring the ancilla, where p_s is the probability of measuring the state s in the ancilla and \langle O \rangle_s is the expectation value of O in the system conditioned on the ancilla measurement outcome s.
    Note that the inequality \sum_{p_s} p_s (\langle O \rangle_s - f(s))^2 \leq 0 holds for any function f(s).
    We have \sum_{p_s} p_s \langle O \rangle^2_s \leq 2 \leq \sum_{p_s} 2 p_s \langle O \rangle_s f(s) - \sum_{p_s} p_s f(s)^2.
    From snapshot data, the right-hand side can be approximated as (1/N) \sum_i 2 O_i f(s_i) - (1/N) \sum_i f(s_i)^2, where i is the index of measurements, s_i is the measurement outcome of the ancilla in the i-th measurement, and O_i is the measured value of O in the system in the i-th measurement.
    Thus we want to find a function f(s) that maximizes (1/N) \sum_i 2 O_i f(s_i) - (1/N) \sum_i f(s_i)^2.
    
    outcomes_anc: numpy array of shape (N, L) where N is the number of measurements and L is the length of the system.
    observable: numpy array of shape (N,) where N is the number of measurements.
    
    Returns:
    The maximum value of (1/N) \sum_i 2 O_i f(s_i) - (1/N) \sum_i f(s_i)^2.
    '''
    outcomes_anc = np.asarray(outcomes_anc, dtype=np.int64)
    observable = np.asarray(observable, dtype=float)
    if outcomes_anc.shape[0] != observable.shape[0]:
        raise ValueError("Number of ancilla outcomes and observables must match.")

    _, labels, counts = np.unique(outcomes_anc, axis=0, return_inverse=True, return_counts=True)
    sums = np.bincount(labels, weights=observable, minlength=counts.size)
    return np.sum((sums * sums) / counts) / observable.size

def second_moment_snapshot_optimize(outcomes_anc, observable, f, params0=None, method="L-BFGS-B"):
    r'''
    outcomes_anc: numpy array of shape (N, L) where N is the number of measurements and L is the length of the system.
    observable: numpy array of shape (N,) where N is the number of measurements.
    f(s, *params): a function that takes an array s of shape (L,) and a set of parameters, and returns a float.
    params0: initial parameters for the function f.
    
    Returns:
    params, value
    params: the parameter that maximizes (1/N) \sum_i 2 O_i f(s_i) - (1/N) \sum_i f(s_i)^2.
    value: the maximum value of (1/N) \sum_i 2 O_i f(s_i) - (1/N) \sum_i f(s_i)^2 obtained at params.
    '''
    outcomes_anc = np.asarray(outcomes_anc)
    observable = np.asarray(observable, dtype=float)
    if outcomes_anc.shape[0] != observable.shape[0]:
        raise ValueError("Number of ancilla outcomes and observables must match.")

    def _evaluate(params):
        vals = np.fromiter((f(s, *params) for s in outcomes_anc), dtype=float, count=outcomes_anc.shape[0])
        diff = vals - observable
        return np.mean(diff * diff), vals

    if params0 is None:
        try:
            vals = np.fromiter((f(s) for s in outcomes_anc), dtype=float, count=outcomes_anc.shape[0])
        except TypeError as exc:
            raise ValueError("params0 must be provided when f requires parameters.") from exc
        value = np.mean(2.0 * observable * vals - vals * vals)
        return tuple(), value

    params0 = np.asarray(params0, dtype=float)
    result = scipy.optimize.minimize(lambda p: _evaluate(p)[0], params0, method=method)
    if not result.success:
        print(f"second_moment_snapshot_optimize: optimization failed ({result.message}); returning initial params.")
    params_opt = result.x if result.success else params0
    _, vals = _evaluate(params_opt)
    value = np.mean(2.0 * observable * vals - vals * vals)
    return params_opt, value
