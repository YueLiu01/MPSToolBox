import numpy as np
import scipy
import time
import numba as nb
from numba.core.errors import TypingError
import quspin
from utils import POVM

def ground_state_Ising(L, J=1.0, h=1.0, bc='periodic'):
    """
    Compute the ground state energy and wavefunction of the transverse field Ising model
    using exact diagonalization.

    Parameters:
    L (int): Number of spins in the chain.
    J (float): Interaction strength between neighboring spins.
    h (float): Transverse field strength.
    bc (str): Boundary conditions ('periodic' or 'open').

    Returns:
    tuple: Ground state energy and wavefunction.
    """
    if L <= 0:
        raise ValueError("Chain length L must be positive.")

    bc = bc.lower()
    if bc == 'periodic':
        bonds = [(i, (i + 1) % L) for i in range(L)]
    elif bc == 'open':
        bonds = [(i, i + 1) for i in range(L - 1)]
    else:
        raise ValueError("bc must be 'periodic' or 'open'.")

    J_terms = [[-J, i, j] for i, j in bonds]
    h_terms = [[-h, i] for i in range(L)]

    basis = quspin.basis.spin_basis_1d(L)
    H = quspin.operators.hamiltonian(
        [["zz", J_terms], ["x", h_terms]],
        [],
        basis=basis,
        dtype=np.float64,
        check_herm=False,
        check_symm=False,
    )

    E, psi = H.eigsh(k=1, which='SA')

    return float(E[0]), psi[:, 0]

def povm_onsite_measurement(psi, povm_list, site_ind, outcomes):
    '''
    Apply a POVM measurement on the wavefunction `psi` at the specified site indices.
    
    Parameters:
    psi (numpy array): the wavefunction to be measured.
    povm_list (list of POVM objects): list of POVM operators. The POVM class is imported from MPSToolBox.POVM.
    site_ind (list of int): site indices where the POVM measurements are applied. len(site_ind) should be equal to len(povm_list).
    outcomes (list of int): measurement outcomes corresponding to each POVM operator.
    
    Returns:
    psi_measured (numpy array): the wavefunction after the POVM measurement.
    probability (float): the probability of the measurement outcome.
    '''
    
    psi = np.asarray(psi, dtype=complex)
    if psi.ndim != 1:
        psi = psi.reshape(-1)

    n_ops = len(povm_list)
    if n_ops != len(site_ind) or n_ops != len(outcomes):
        raise ValueError("povm_list, site_ind, and outcomes must have the same length.")

    local_dim = povm_list[0].kraus_ops[0].shape[0]
    if local_dim <= 0:
        raise ValueError("Invalid local dimension inferred from POVM.")

    total_dim = psi.size
    L = int(round(np.log(total_dim) / np.log(local_dim)))
    if local_dim ** L != total_dim:
        raise ValueError("Wavefunction size is not compatible with the POVM local dimension.")

    psi_tensor = psi.reshape((local_dim,) * L)

    for povm, site, outcome in zip(povm_list, site_ind, outcomes):
        site_axis = L - 1 - int(site)  # quspin uses little-endian ordering
        outcomes_arr = np.asarray(povm.outcomes)
        match_idx = np.nonzero(outcomes_arr == outcome)[0]
        if match_idx.size == 0:
            raise ValueError(f"Outcome {outcome} not found in POVM outcomes.")
        K = povm.kraus_ops[int(match_idx[0])]

        psi_tensor = np.tensordot(K, psi_tensor, axes=([1], [site_axis]))
        psi_tensor = np.moveaxis(psi_tensor, 0, site_axis)

    probability = float(np.sum(np.abs(psi_tensor) ** 2).real)
    if probability > 0.0:
        psi_measured = (psi_tensor / np.sqrt(probability)).reshape(-1)
    else:
        psi_measured = np.zeros_like(psi)

    return psi_measured, probability

def povm_onsite_sampling(psi, povm_list, site_ind, rng):
    '''
    Sample measurement outcomes from POVM measurements on the wavefunction `psi` at the specified site indices.
    
    Parameters:
    psi (numpy array): the wavefunction to be measured.
    povm_list (list of POVM objects): list of POVM operators.
    site_ind (list of int): site indices where the POVM measurements are applied. len(site_ind) should be equal to len(povm_list).
    rng (np.random.Generator): random number generator for sampling.
    
    Returns:
    outcomes (list of int): sampled measurement outcomes.
    probability (float): the probability of the sampled measurement outcomes.
    '''
    
    psi = np.asarray(psi, dtype=complex)
    if psi.ndim != 1:
        psi = psi.reshape(-1)

    if rng is None:
        rng = np.random.default_rng()

    n_ops = len(povm_list)
    if n_ops != len(site_ind):
        raise ValueError("povm_list and site_ind must have the same length.")

    local_dim = povm_list[0].kraus_ops[0].shape[0]
    total_dim = psi.size
    L = int(round(np.log(total_dim) / np.log(local_dim)))
    if local_dim ** L != total_dim:
        raise ValueError("Wavefunction size is not compatible with the POVM local dimension.")

    psi_tensor = psi.reshape((local_dim,) * L)
    outcomes = []
    total_probability = 1.0

    for povm, site in zip(povm_list, site_ind):
        site_axis = L - 1 - int(site)
        probs = []
        psi_candidates = []
        for K in povm.kraus_ops:
            tmp = np.tensordot(K, psi_tensor, axes=([1], [site_axis]))
            tmp = np.moveaxis(tmp, 0, site_axis)
            prob = float(np.sum(np.abs(tmp) ** 2).real)
            probs.append(prob)
            psi_candidates.append(tmp)

        prob_sum = float(np.sum(probs))
        if prob_sum <= 0.0:
            outcomes.append(povm.outcomes[0])
            psi_tensor = np.zeros_like(psi_tensor)
            total_probability = 0.0
            continue

        probs_norm = np.asarray(probs, dtype=float) / prob_sum
        choice = int(rng.choice(len(probs_norm), p=probs_norm))
        outcomes.append(povm.outcomes[choice])

        selected_prob = probs[choice]
        total_probability *= probs_norm[choice]

        if selected_prob > 0.0:
            psi_tensor = psi_candidates[choice] / np.sqrt(selected_prob)
        else:
            psi_tensor = np.zeros_like(psi_tensor)

    return outcomes, float(total_probability)

@nb.njit(cache=True, fastmath=True)
def _norm_squared(arr):
    flat = arr.ravel()
    acc = 0.0
    for i in range(flat.size):
        v = flat[i]
        acc += v.real * v.real + v.imag * v.imag
    return acc

@nb.njit(cache=True, fastmath=True)
def _vdot_conj(a, b):
    flat_a = a.ravel()
    flat_b = b.ravel()
    real = 0.0
    imag = 0.0
    for i in range(flat_a.size):
        va = flat_a[i]
        vb = flat_b[i]
        real += va.real * vb.real + va.imag * vb.imag
        imag += va.real * vb.imag - va.imag * vb.real
    return real + 1j * imag

def measurement_altered_moments(psi, k, povm_list, site_ind, ops):
    r'''
    Compute measurement-altered moments on the wavefunction `psi` after applying POVM measurements.
    Let s be all possible measurement outcomes, then the measurement-altered moment is defined as
    \sum_s P(s) (<O>_s)^2, where P(s) is the probability of measurement outcome s, and (<O>_s)^k is the k-th moment of operator O on the post-measurement wavefunction after outcome s. We sum over all possible measurement outcomes s. The operator O is a product of onsite operators represented as a list of tuples (op_matrix, site_index), where op_matrix is a numpy array and site_index is an integer. For example O can be a two-point correlation function Z_i Z_j represented as [(sZ, i), (sZ, j)] or a single-site operator Z_i represented as [(sZ, i)].
    
    Parameters:
    psi (numpy array): the wavefunction.
    k (float): moments power.
    povm_list (list of POVM objects): list of POVM operators.
    site_ind (list of int): site indices where the POVM measurements are applied. len(site_ind) should be equal to len(povm_list).
    ops (list of tuples): A list of onsite operators to compute the expectation value. Each onsite operator is represented as a tuple (op_matrix, site_index).
    
    Returns:
    result (float): the measurement-altered moment \sum_s P(s) (<O>_s)^2.
    '''
    psi = np.asarray(psi, dtype=complex)
    if psi.ndim != 1:
        psi = psi.reshape(-1)

    total_dim = psi.size
    if len(povm_list) != len(site_ind):
        raise ValueError("povm_list and site_ind must have the same length.")

    if povm_list:
        local_dim = povm_list[0].kraus_ops[0].shape[0]
    elif ops:
        local_dim = np.asarray(ops[0][0], dtype=complex).shape[0]
    else:
        local_dim = 2
        if total_dim:
            L_guess = int(round(np.log(total_dim) / np.log(local_dim)))
            if local_dim ** L_guess != total_dim:
                raise ValueError("Cannot infer local dimension from inputs.")

    if local_dim <= 0:
        raise ValueError("Invalid local dimension inferred from POVM.")

    L = int(round(np.log(total_dim) / np.log(local_dim))) if total_dim else 0
    if local_dim ** L != total_dim:
        raise ValueError("Wavefunction size is not compatible with the POVM local dimension.")

    site_axes = [L - 1 - int(s) for s in site_ind]
    for ax in site_axes:
        if ax < 0 or ax >= L:
            raise ValueError("site_ind contains an invalid site index.")

    meas_data = [(tuple(povm.kraus_ops), ax) for povm, ax in zip(povm_list, site_axes)]

    op_terms = []
    for op_mat, site in ops:
        op_arr = np.asarray(op_mat, dtype=complex)
        if op_arr.shape != (local_dim, local_dim):
            raise ValueError("Each operator must match the local dimension.")
        axis = L - 1 - int(site)
        if axis < 0 or axis >= L:
            raise ValueError("Operator site index out of range.")
        op_terms.append((op_arr, axis))

    psi_tensor = psi.reshape((local_dim,) * L)
    k = float(k)

    def apply_on_axis(state_tensor, op_mat, axis):
        res = np.tensordot(op_mat, state_tensor, axes=([1], [axis]))
        return np.moveaxis(res, 0, axis)

    def apply_ops(state_tensor):
        res = state_tensor
        for op_mat, axis in op_terms:
            res = apply_on_axis(res, op_mat, axis)
        return res

    result = 0.0
    stack = [(0, psi_tensor)]
    meas_len = len(meas_data)

    while stack:
        depth, state_tensor = stack.pop()
        if depth == meas_len:
            prob = _norm_squared(state_tensor)
            if prob <= 0.0:
                continue

            if op_terms:
                op_state = apply_ops(state_tensor)
                exp_num = _vdot_conj(state_tensor, op_state)
                exp_val = exp_num / prob
                contrib = prob * (exp_val.real ** k)
            else:
                contrib = prob * (1.0 ** k)

            result += contrib
            continue

        kraus_list, axis = meas_data[depth]
        for K in kraus_list:
            new_state = apply_on_axis(state_tensor, K, axis)
            stack.append((depth + 1, new_state))

    return float(result)

# add a test code for L=10 critical ising model
if __name__ == "__main__":
    L = 10
    J = 1.0
    h = 1.0
    energy, wavefunction = ground_state_Ising(L, J, h, bc='periodic')
    print(f"Ground state energy for L={L}, J={J}, h={h}: {energy}")
    print(f"Ground state wavefunction (first 10 components): {wavefunction}")

    from utils import weak_measurement_pauli, sZ

    beta = 0.3
    povm = weak_measurement_pauli(sZ, beta=beta, real=True)

    ops = [(sZ, 0)]
    povm_list = [povm] * L
    site_ind = np.arange(L)
    altered_moment = measurement_altered_moments(wavefunction, k=2, povm_list=povm_list, site_ind=site_ind, ops=ops)
    print(f"Measurement-altered second moment: {altered_moment}")
