import numpy as np
import scipy

# Basic Pauli matrices and identity
sX = np.array([[0, 1.0], [1.0, 0]])
sY = np.array([[0, -1.0j], [1.0j, 0]])
sZ = np.array([[1.0, 0], [0, -1.0]])
Id = np.array([[1.0, 0], [0, 1.0]])

# Common scipy helpers
expm = scipy.linalg.expm
curvefit = scipy.optimize.curve_fit


class POVM:
    '''
    POVM class with properties:
    kraus_ops: list of kraus operators as numpy arrays
    outcomes: list of measurement outcomes corresponding to kraus operators
    '''
    def __init__(self, kraus_ops, outcomes):
        self.kraus_ops = kraus_ops
        self.outcomes = outcomes
        # check povm validity
        dim = kraus_ops[0].shape[0]
        identity = np.zeros((dim, dim), dtype=complex)
        for K in kraus_ops:
            identity += K.conj().T @ K
        if not np.allclose(identity, np.eye(dim), atol=1e-10):
            raise ValueError("Invalid POVM: sum of K_i^† K_i does not equal identity.")


def weak_measurement_pauli(op: np.ndarray, beta, real: bool = False) -> POVM:
    '''
    Construct weak measurement POVM object for a given Pauli operator `op` and measurement strength `beta`.
    The Kraus operators are proportional to exp(beta * op) and exp(-beta * op).
    op: numpy array, Pauli operator (e.g., sX, sY, sZ or their combinations)
    beta: float, measurement strength
    real: bool, if True, return only the real part of the Kraus operators. (This can make sampling faster.)
    Return: POVM object
    '''
    op = np.asarray(op, dtype=complex)
    if op.shape[0] != op.shape[1]:
        raise ValueError("Operator must be square.")
    dim = op.shape[0]
    identity = np.eye(dim, dtype=complex)
    if not np.allclose(op @ op, identity, atol=1e-10):
        raise ValueError("Weak measurement requires a Pauli-like operator with op^2 = I.")
    beta = float(beta)
    abs_beta = abs(beta)
    exp_neg2b = np.exp(-2.0 * abs_beta)
    denom = np.sqrt(1.0 + exp_neg2b * exp_neg2b)  # sqrt(1 + exp(-4|beta|))
    a = 0.5 * (1.0 + exp_neg2b) / denom  # norm * cosh(beta)
    s = 1.0 if beta >= 0.0 else -1.0
    one_minus = -np.expm1(-2.0 * abs_beta)         # 1 - exp(-2|beta|)
    c = 0.5 * s * one_minus / denom                # norm * sinh(beta)
    K_plus = a * identity + c * op
    K_minus = a * identity - c * op
    if real:
        K_plus = K_plus.real
        K_minus = K_minus.real
    return POVM(kraus_ops=[K_minus, K_plus], outcomes=[-1, +1])
