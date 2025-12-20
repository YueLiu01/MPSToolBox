import numpy as np
import scipy

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

def f_window_average_hardcut(s, center, length_scale, amplitude, periodic=True):
    r'''
    s: ndarray of shape (L,) where L is the length of the system.
    center: center of the window
    length_scale: length scale of the window, the size of the window is 2*length_scale
    amplitude: overall prefactor multiplier
    periodic: whether to use periodic boundary conditions
    
    Returns:
    The window-averaged value of s at center.
    '''
    s = np.asarray(s, dtype=float)
    L = s.shape[0]
    if L == 0:
        return 0.0

    c = int(round(center))
    w = int(length_scale)
    if w <= 0:
        return 0.0

    if periodic:
        idx = (np.arange(c - w, c + w) % L)
        window = s[idx]
    else:
        start = max(0, c - w)
        end = min(L, c + w)
        if end <= start:
            return 0.0
        window = s[start:end]

    return amplitude * np.mean(window)

def f_window_average_softcut(s, center, length_scale, amplitude, periodic=True):
    r'''
    s: ndarray of shape (L,) where L is the length of the system.
    center: center of the window
    length_scale: length scale of the window, the size of the window is 2*length_scale
    amplitude: overall prefactor multiplier
    periodic: whether to use periodic boundary conditions
    
    Returns:
    The window-averaged value of s at center. The weights decay as a gaussian function away from the center with standard deviation length_scale.
    '''
    s = np.asarray(s, dtype=float)
    L = s.shape[0]
    if L == 0:
        return 0.0

    sigma = float(length_scale)
    if sigma <= 0.0:
        return 0.0

    center = float(center)
    idx = np.arange(L, dtype=float)
    if periodic:
        # minimal distance on a ring preserves periodicity
        dist = np.abs(((idx - center + 0.5 * L) % L) - 0.5 * L)
    else:
        dist = np.abs(idx - center)

    weights = np.exp(-0.5 * (dist / sigma) ** 2)
    norm = weights.sum()
    if norm == 0.0:
        return 0.0

    return amplitude * np.dot(s, weights) / norm
    
'''
Testing area:
'''
# import matplotlib.pyplot as plt
# L = 10
# for beta in [0.3]:
#     outcomes_sys = np.load('data/CritIsing_L'+str(L)+'_beta'+str(beta)+'_Zoutcomes_sys.npy')
#     outcomes_anc = np.load('data/CritIsing_L'+str(L)+'_beta'+str(beta)+'_Zoutcomes_anc.npy')
#     _, N = outcomes_sys.shape
#     N_list = np.logspace(1, 5, 30).astype(int)
#     N_shuffle = 100
#     results = []
#     for n in N_list:
#         result_i = []
#         for i in range(L):
#             for _ in range(N_shuffle):
#                 observable = outcomes_sys[:,i]
                
#                 # shuffle outcomes_anc together with observable
#                 indices = np.random.permutation(N)
                
#                 s = outcomes_anc[indices,:][:n]
#                 o = observable[indices][:n]
                
#                 def myfun(s, length_scale, amplitude):
#                     return f_window_average_hardcut(s, i, length_scale, amplitude, periodic=True)
                
#                 result_i.append(second_moment_snapshot_optimize(s, o, f=myfun, params0=[L/4, 1.0])[1])
#         results.append(np.mean(result_i))
#     plt.semilogx(N_list, results)
