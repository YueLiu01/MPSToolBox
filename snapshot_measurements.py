import numpy as np
import scipy
import time
import numba as nb
from numba.core.errors import TypingError
from sklearn.linear_model import LinearRegression

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
    success = True
    if not result.success:
        success = False
        print(f"second_moment_snapshot_optimize: optimization failed ({result.message}); returning initial params.")
    params_opt = result.x if result.success else params0
    _, vals = _evaluate(params_opt)
    value = np.mean(2.0 * observable * vals - vals * vals)
    return params_opt, value, success

def second_moment_snapshot_optimize_withgrad(outcomes_anc, observable, f_with_grad, params0, method="L-BFGS-B"):
    r'''
    Like second_moment_snapshot_optimize but expects f_with_grad(s, params) -> (value, grad) and uses the gradient to speed up optimization.
    '''
    outcomes_anc = np.asarray(outcomes_anc)
    observable = np.asarray(observable, dtype=float)
    if outcomes_anc.shape[0] != observable.shape[0]:
        raise ValueError("Number of ancilla outcomes and observables must match.")

    params0 = np.asarray(params0, dtype=float)

    def _loss_and_grad(p):
        loss = 0.0
        grad = np.zeros_like(p)
        for s, o in zip(outcomes_anc, observable):
            val, g = f_with_grad(s, p)
            diff = val - o
            loss += diff * diff
            grad += 2.0 * diff * g
        n = observable.size
        return loss / n, grad / n

    result = scipy.optimize.minimize(_loss_and_grad, params0, method=method, jac=True)
    success = bool(result.success)
    params_opt = result.x if success else params0

    vals = np.fromiter((f_with_grad(s, params_opt)[0] for s in outcomes_anc), dtype=float, count=outcomes_anc.shape[0])
    value = np.mean(2.0 * observable * vals - vals * vals)
    return params_opt, value, success

def second_moment_snapshot_optimize_SGD(outcomes_anc, observable, f, params0=None, epochs=10, lr=0.01, batch_size=32, rng_seed=None):
    r'''
    Similar to second_moment_snapshot_optimize, but using stochastic gradient descent (SGD) to optimize the parameters.
    '''
    outcomes_anc = np.asarray(outcomes_anc)
    observable = np.asarray(observable, dtype=float)
    if outcomes_anc.shape[0] != observable.shape[0]:
        raise ValueError("Number of ancilla outcomes and observables must match.")

    if params0 is None:
        try:
            vals = np.fromiter((f(s) for s in outcomes_anc), dtype=float, count=outcomes_anc.shape[0])
        except TypeError as exc:
            raise ValueError("params0 must be provided when f requires parameters.") from exc
        value = np.mean(2.0 * observable * vals - vals * vals)
        return tuple(), value, True

    rng = np.random.default_rng(rng_seed)
    params = np.asarray(params0, dtype=float).copy()
    n_samples = outcomes_anc.shape[0]
    batch_size = max(1, min(int(batch_size), n_samples))
    epochs = int(epochs)
    lr = float(lr)

    def _eval_batch(p, s_batch):
        return np.fromiter((f(s, *p) for s in s_batch), dtype=float, count=s_batch.shape[0])

    for _ in range(epochs):
        indices = rng.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            idx = indices[start:start + batch_size]
            s_batch = outcomes_anc[idx]
            o_batch = observable[idx]

            preds = _eval_batch(params, s_batch)
            diff = preds - o_batch

            grads = np.empty_like(params)
            for j in range(params.size):
                orig = params[j]
                eps = 1e-5 * max(1.0, abs(orig))

                params[j] = orig + eps
                preds_plus = _eval_batch(params, s_batch)

                params[j] = orig - eps
                preds_minus = _eval_batch(params, s_batch)

                params[j] = orig
                deriv = (preds_plus - preds_minus) * (0.5 / eps)
                grads[j] = np.mean(2.0 * diff * deriv)

            params -= lr * grads

    final_vals = _eval_batch(params, outcomes_anc)
    value = np.mean(2.0 * observable * final_vals - final_vals * final_vals)
    return params, value, True



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

def f_window_average_softcut_with_grad(s, center, length_scale, amplitude, periodic=True):
    r'''
    Returns value and gradient with respect to (length_scale, amplitude). center and periodic are treated as constants.
    '''
    s = np.asarray(s, dtype=float)
    L = s.shape[0]
    if L == 0:
        return 0.0, np.zeros(2)

    sigma = float(length_scale)
    if sigma <= 0.0:
        return 0.0, np.zeros(2)

    center = float(center)
    idx = np.arange(L, dtype=float)
    if periodic:
        delta = ((idx - center + 0.5 * L) % L) - 0.5 * L
    else:
        delta = idx - center

    dist = np.abs(delta)
    weights = np.exp(-0.5 * (dist / sigma) ** 2)
    norm = weights.sum()
    if norm == 0.0:
        return 0.0, np.zeros(2)

    num = np.dot(s, weights)
    val0 = num / norm
    val = amplitude * val0

    dw_sigma = weights * (dist * dist) / (sigma ** 3)

    s_dw_sigma = np.dot(s, dw_sigma)
    sum_dw_sigma = dw_sigma.sum()

    grad_sigma = amplitude * (s_dw_sigma * norm - num * sum_dw_sigma) / (norm * norm)
    grad_amp = val0
    return val, np.array([grad_sigma, grad_amp])

@nb.njit
def f_window_average_hardcut_njit(s, center, length_scale, amplitude, periodic):
    L = s.shape[0]
    if L == 0:
        return 0.0

    c = int(center)
    w = int(length_scale)
    if w <= 0:
        return 0.0

    total = 0.0
    count = 0
    if periodic:
        for offset in range(-w, w):
            idx = (c + offset) % L
            total += s[idx]
            count += 1
    else:
        start = max(0, c - w)
        end = min(L, c + w)
        for idx in range(start, end):
            total += s[idx]
            count += 1

    if count == 0:
        return 0.0
    return amplitude * (total / count)

@nb.njit
def f_window_average_softcut_njit(s, center, length_scale, amplitude, periodic):
    L = s.shape[0]
    if L == 0:
        return 0.0

    sigma = float(length_scale)
    if sigma <= 0.0:
        return 0.0

    center_f = float(center)
    half_L = 0.5 * L
    num = 0.0
    den = 0.0
    for idx in range(L):
        dist = float(idx) - center_f
        if periodic:
            dist = abs(((dist + half_L) % L) - half_L)
        else:
            dist = abs(dist)
        w = np.exp(-0.5 * (dist / sigma) ** 2)
        den += w
        num += w * s[idx]

    if den == 0.0:
        return 0.0
    return amplitude * (num / den)

@nb.njit
def f_window_average_softcut_njit_with_grad(s, center, length_scale, amplitude, periodic):
    L = s.shape[0]
    if L == 0:
        return 0.0, np.zeros(2)

    sigma = float(length_scale)
    if sigma <= 0.0:
        return 0.0, np.zeros(2)

    center_f = float(center)
    half_L = 0.5 * L

    num = 0.0
    den = 0.0
    s_dw_sigma = 0.0
    sum_dw_sigma = 0.0

    for idx in range(L):
        delta = float(idx) - center_f
        if periodic:
            delta = ((delta + half_L) % L) - half_L

        dist = delta if delta >= 0.0 else -delta

        exp_arg = -0.5 * (dist / sigma) ** 2
        w = np.exp(exp_arg)
        num += s[idx] * w
        den += w

        dw_sigma = w * (dist * dist) / (sigma * sigma * sigma)

        s_dw_sigma += s[idx] * dw_sigma
        sum_dw_sigma += dw_sigma

    if den == 0.0:
        return 0.0, np.zeros(2)

    val0 = num / den
    val = amplitude * val0

    grad_sigma = amplitude * (s_dw_sigma * den - num * sum_dw_sigma) / (den * den)
    grad_amp = val0

    g = np.empty(2)
    g[0] = grad_sigma
    g[1] = grad_amp
    return val, g

@nb.njit
def f_window_average_softcut_njit_with_hessian(s, center, length_scale, amplitude, periodic):
    L = s.shape[0]
    if L == 0:
        return 0.0, np.zeros(2), np.zeros((2, 2))

    sigma = float(length_scale)
    if sigma <= 0.0:
        return 0.0, np.zeros(2), np.zeros((2, 2))

    center_f = float(center)
    half_L = 0.5 * L

    num = 0.0
    den = 0.0
    s_dw_sigma = 0.0
    sum_dw_sigma = 0.0
    s_d2w = 0.0
    sum_d2w = 0.0

    for idx in range(L):
        delta = float(idx) - center_f
        if periodic:
            delta = ((delta + half_L) % L) - half_L

        dist = delta if delta >= 0.0 else -delta
        dist2 = dist * dist

        exp_arg = -0.5 * (dist / sigma) ** 2
        w = np.exp(exp_arg)
        num += s[idx] * w
        den += w

        dw_sigma = w * dist2 / (sigma * sigma * sigma)
        d2w = w * (dist2 * dist2 / (sigma ** 6) - 3.0 * dist2 / (sigma ** 4))

        s_dw_sigma += s[idx] * dw_sigma
        sum_dw_sigma += dw_sigma
        s_d2w += s[idx] * d2w
        sum_d2w += d2w

    if den == 0.0:
        return 0.0, np.zeros(2), np.zeros((2, 2))

    inv_den = 1.0 / den
    val0 = num * inv_den
    val = amplitude * val0

    num_term = s_dw_sigma * den - num * sum_dw_sigma
    grad_sigma = amplitude * num_term * inv_den * inv_den
    grad_amp = val0

    h_sigma_sigma = amplitude * ((s_d2w * den - num * sum_d2w) * inv_den * inv_den - 2.0 * num_term * sum_dw_sigma * inv_den * inv_den * inv_den)
    h_sigma_amp = num_term * inv_den * inv_den
    h_amp_sigma = h_sigma_amp
    h_amp_amp = 0.0

    grad = np.empty(2)
    grad[0] = grad_sigma
    grad[1] = grad_amp

    hess = np.empty((2, 2))
    hess[0, 0] = h_sigma_sigma
    hess[0, 1] = h_sigma_amp
    hess[1, 0] = h_amp_sigma
    hess[1, 1] = h_amp_amp
    return val, grad, hess

@nb.njit
def _njit_loss_grad_with_params(outcomes_anc, observable, params, f_njit_with_grad):
    n = observable.shape[0]
    grad = np.zeros(params.shape[0])
    loss = 0.0
    for i in range(n):
        val, g = f_njit_with_grad(outcomes_anc[i], params)
        diff = val - observable[i]
        loss += diff * diff
        for j in range(params.shape[0]):
            grad[j] += 2.0 * diff * g[j]
    return loss / n, grad / n

@nb.njit
def _njit_loss_grad_hess_with_params(outcomes_anc, observable, params, f_njit_with_hessian):
    n = observable.shape[0]
    grad = np.zeros(params.shape[0])
    hess = np.zeros((params.shape[0], params.shape[0]))
    loss = 0.0
    for i in range(n):
        val, g, h = f_njit_with_hessian(outcomes_anc[i], params)
        diff = val - observable[i]
        loss += diff * diff
        for j in range(params.shape[0]):
            grad[j] += 2.0 * diff * g[j]
            for k in range(params.shape[0]):
                hess[j, k] += 2.0 * (g[j] * g[k] + diff * h[j, k])
    inv_n = 1.0 / n
    grad *= inv_n
    hess *= inv_n
    return loss * inv_n, grad, hess

@nb.njit
def _njit_loss_with_params(outcomes_anc, observable, params, f_njit):
    total = 0.0
    n = observable.shape[0]
    for i in range(n):
        val = f_njit(outcomes_anc[i], params)
        diff = val - observable[i]
        total += diff * diff
    return total / n

@nb.njit
def _njit_predict_with_params(outcomes_anc, params, f_njit):
    n = outcomes_anc.shape[0]
    vals = np.empty(n)
    for i in range(n):
        vals[i] = f_njit(outcomes_anc[i], params)
    return vals

@nb.njit
def _njit_predict_no_params(outcomes_anc, f_njit):
    n = outcomes_anc.shape[0]
    vals = np.empty(n)
    for i in range(n):
        vals[i] = f_njit(outcomes_anc[i])
    return vals

def second_moment_snapshot_optimize_njit(outcomes_anc, observable, f_njit, params0=None, method="L-BFGS-B"):
    r'''
    Similar to second_moment_snapshot_optimize but expects a numba-jitted function f_njit for faster evaluation.
    
    f_njit: numba.njit-ed callable. If params0 is provided, it should have signature f_njit(s, params).
            If params0 is None, it should have signature f_njit(s).
    '''
    outcomes_anc = np.ascontiguousarray(outcomes_anc)
    observable = np.ascontiguousarray(observable, dtype=float)
    if outcomes_anc.shape[0] != observable.shape[0]:
        raise ValueError("Number of ancilla outcomes and observables must match.")

    if params0 is None:
        try:
            vals = _njit_predict_no_params(outcomes_anc, f_njit)
        except (TypeError, TypingError) as exc:
            raise ValueError("params0 must be provided when f_njit requires parameters.") from exc
        value = np.mean(2.0 * observable * vals - vals * vals)
        return tuple(), value

    params0 = np.ascontiguousarray(params0, dtype=float)

    def _objective(p):
        return float(_njit_loss_with_params(outcomes_anc, observable, p, f_njit))

    result = scipy.optimize.minimize(_objective, params0, method=method)
    success = bool(result.success)
    params_opt = result.x if success else params0

    vals = _njit_predict_with_params(outcomes_anc, params_opt, f_njit)
    value = np.mean(2.0 * observable * vals - vals * vals)
    return params_opt, value, success

def second_moment_snapshot_optimize_njit_withgrad(outcomes_anc, observable, f_njit_with_grad, params0, method="L-BFGS-B"):
    r'''
    Numba-accelerated optimizer that uses gradients. f_njit_with_grad(s, params) should return (value, grad).
    '''
    outcomes_anc = np.ascontiguousarray(outcomes_anc)
    observable = np.ascontiguousarray(observable, dtype=float)
    if outcomes_anc.shape[0] != observable.shape[0]:
        raise ValueError("Number of ancilla outcomes and observables must match.")

    params0 = np.ascontiguousarray(params0, dtype=float)

    def _objective(p):
        loss, grad = _njit_loss_grad_with_params(outcomes_anc, observable, p, f_njit_with_grad)
        return float(loss), np.array(grad, dtype=float)

    result = scipy.optimize.minimize(_objective, params0, method=method, jac=True)
    success = bool(result.success)
    params_opt = result.x if success else params0

    vals = np.fromiter((f_njit_with_grad(s, params_opt)[0] for s in outcomes_anc), dtype=float, count=outcomes_anc.shape[0])
    value = np.mean(2.0 * observable * vals - vals * vals)
    return params_opt, value, success

def second_moment_snapshot_optimize_njit_withhessian(outcomes_anc, observable, f_njit_with_hessian, params0, method="trust-ncg"):
    r'''
    Numba-accelerated optimizer that uses gradients and Hessians.
    f_njit_with_hessian(s, params) should return (value, grad, hess).
    '''
    outcomes_anc = np.ascontiguousarray(outcomes_anc)
    observable = np.ascontiguousarray(observable, dtype=float)
    if outcomes_anc.shape[0] != observable.shape[0]:
        raise ValueError("Number of ancilla outcomes and observables must match.")

    params0 = np.ascontiguousarray(params0, dtype=float)

    def _objective(p):
        loss, grad, hess = _njit_loss_grad_hess_with_params(outcomes_anc, observable, p, f_njit_with_hessian)
        return float(loss), np.array(grad, dtype=float), np.array(hess, dtype=float)

    result = scipy.optimize.minimize(lambda x: _objective(x)[0],
                                     params0,
                                     method=method,
                                     jac=lambda x: _objective(x)[1],
                                     hess=lambda x: _objective(x)[2])
    success = bool(result.success)
    params_opt = result.x if success else params0

    vals = np.fromiter((f_njit_with_hessian(s, params_opt)[0] for s in outcomes_anc), dtype=float, count=outcomes_anc.shape[0])
    value = np.mean(2.0 * observable * vals - vals * vals)
    return params_opt, value, success
    
    
def second_moment_snapshot_linear_regression(outcomes_anc, observable, center=0, periodic=True, enforce_reflection_symmetry=False, bias=True):
    r'''
    In this function we take f(s) to be a linear combination of components of s, i.e., f(s) = b + \sum_j a_j s_j where s_j is the j-th component of s. And a_j and b are the parameters to be optimized. Note that the target function (1/N) \sum_i 2 O_i f(s_i) - (1/N) \sum_i f(s_i)^2 can be rewritten as - (1/N) \sum_i (f(s_i) - O_i)^2 + (1/N) \sum_i O_i^2. Thus maximizing the target function is equivalent to minimizing the mean squared error between f(s_i) and O_i.
    
    outcomes_anc: numpy array of shape (N, L) where N is the number of measurements and L is the length of the system.
    observable: numpy array of shape (N,) where N is the number of measurements.
    center (int): the index of the center of the observable.
    periodic (bool): whether to use periodic boundary conditions.
    enforce_reflection_symmetry (bool): whether to enforce reflection symmetry on the parameters. If True, a_{center + d} = a_{center - d} (note that if periodic=True, the reflection symmetry should be considered with periodic boundary conditions).
    bias (bool): whether to include a constant term b in the linear combination.
    
    Returns:
    params, value
    params: the parameters a_j that maximizes (1/N) \sum_i 2 O_i f(s_i) - (1/N) \sum_i f(s_i)^2. It should be a numpy array of shape (L,) (if bias=True, the last entry corresponds to the bias term). If there are multiple optimal solutions, one of them is returned.
    value: the maximum value of (1/N) \sum_i 2 O_i f(s_i) - (1/N) \sum_i f(s_i)^2 obtained at params. It can be 0 if the number of measurements is less than the number of parameters.
    '''
    outcomes_anc = np.asarray(outcomes_anc, dtype=float)
    observable = np.asarray(observable, dtype=float)
    if outcomes_anc.shape[0] != observable.shape[0]:
        raise ValueError("Number of ancilla outcomes and observables must match.")

    n_samples, length = outcomes_anc.shape

    if enforce_reflection_symmetry:
        center_idx = int(center)
        group_id = np.full(length, -1, dtype=int)
        groups = []
        for j in range(length):
            if group_id[j] != -1:
                continue
            partner = (2 * center_idx - j) % length if periodic else (2 * center_idx - j)
            if not periodic and (partner < 0 or partner >= length):
                partner = j
            if group_id[partner] != -1:
                group_id[j] = group_id[partner]
                continue
            gid = len(groups)
            if partner == j:
                groups.append((j,))
            else:
                groups.append((j, partner))
            group_id[j] = gid
            if partner != j:
                group_id[partner] = gid

        n_features = len(groups)
        X = np.empty((n_samples, n_features), dtype=float)
        for gid, idxs in enumerate(groups):
            if len(idxs) == 1:
                X[:, gid] = outcomes_anc[:, idxs[0]]
            else:
                X[:, gid] = outcomes_anc[:, idxs].sum(axis=1)
    else:
        X = outcomes_anc.astype(float, copy=False)
        n_features = length

    if bias:
        X = np.hstack((X, np.ones((n_samples, 1), dtype=float)))

    coef, _, _, _ = np.linalg.lstsq(X, observable, rcond=None)
    preds = X @ coef
    value = np.mean(2.0 * observable * preds - preds * preds)

    if enforce_reflection_symmetry:
        params = np.empty(length, dtype=float)
        for j in range(length):
            params[j] = coef[group_id[j]]
    else:
        params = coef[:length]

    if bias:
        params = np.concatenate((params, np.array([coef[-1]])))
    return params, value


def second_moment_snapshot_linear_regression_sklearn(outcomes_anc, observable, center=0, periodic=True, enforce_reflection_symmetry=False, bias=True):
    r'''
    Same as second_moment_snapshot_linear_regression but uses sklearn's LinearRegression for fitting.
    '''

    outcomes_anc = np.asarray(outcomes_anc, dtype=float)
    observable = np.asarray(observable, dtype=float)
    if outcomes_anc.shape[0] != observable.shape[0]:
        raise ValueError("Number of ancilla outcomes and observables must match.")

    n_samples, length = outcomes_anc.shape

    if enforce_reflection_symmetry:
        center_idx = int(center)
        group_id = np.full(length, -1, dtype=int)
        groups = []
        for j in range(length):
            if group_id[j] != -1:
                continue
            partner = (2 * center_idx - j) % length if periodic else (2 * center_idx - j)
            if not periodic and (partner < 0 or partner >= length):
                partner = j
            if group_id[partner] != -1:
                group_id[j] = group_id[partner]
                continue
            gid = len(groups)
            if partner == j:
                groups.append((j,))
            else:
                groups.append((j, partner))
            group_id[j] = gid
            if partner != j:
                group_id[partner] = gid

        n_features = len(groups)
        X = np.empty((n_samples, n_features), dtype=float)
        for gid, idxs in enumerate(groups):
            if len(idxs) == 1:
                X[:, gid] = outcomes_anc[:, idxs[0]]
            else:
                X[:, gid] = outcomes_anc[:, idxs].sum(axis=1)
    else:
        X = outcomes_anc.astype(float, copy=False)

    model = LinearRegression(fit_intercept=bias)
    model.fit(X, observable)
    preds = model.predict(X)
    value = np.mean(2.0 * observable * preds - preds * preds)

    coef = model.coef_
    if enforce_reflection_symmetry:
        params = np.empty(length, dtype=float)
        for j in range(length):
            params[j] = coef[group_id[j]]
    else:
        params = coef[:length]

    if bias:
        params = np.concatenate((params, np.array([model.intercept_])))
    return params, value


'''
Testing area:
'''
