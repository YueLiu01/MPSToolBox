import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
np.set_printoptions(precision=5, suppress=True, linewidth=100)
plt.rcParams['figure.dpi'] = 150
import tenpy
import tenpy.linalg.np_conserved as npc
from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO
from tenpy.networks.terms import TermList
from tenpy.models.tf_ising import TFIChain
from tenpy.networks.site import SpinSite, SpinHalfSite, kron
from tenpy.networks.mps import TransferMatrix
from tenpy.models.model import CouplingModel, MPOModel, CouplingMPOModel, NearestNeighborModel
from tenpy.models.lattice import Chain, Lattice
tenpy.tools.misc.setup_logging(to_stdout="INFO")

import pickle

sX = np.array([[0, 1.0], [1.0, 0]])
sY = np.array([[0, -1.0j], [1.0j, 0]])
sZ = np.array([[1.0, 0], [0, -1.0]])
Id = np.array([[1.0, 0], [0, 1.0]])

expm = scipy.linalg.expm
curvefit = scipy.optimize.curve_fit

'''
Save and Load
'''
def save_pkl(psi, filename):
    '''
        psi: MPS (or others) to save
        filename: str, file name
    '''
    with open(filename, 'wb') as f:
        pickle.dump(psi, f)

def load_pkl(filename):
    '''
        filename: str, file name
        Return: data in pickle file
    '''
    with open(filename, 'rb') as f:
        psi = pickle.load(f)
    return psi



'''
Models: Critical Ising, Dual Critical Ising, XY, SymmZXZ
'''
class IsingModel(CouplingMPOModel):
    def init_sites(self, model_params):
        return tenpy.networks.site.SpinHalfSite([])
    def init_terms(self, model_params):
        J = model_params.get('J', 1.)
        h = model_params.get('h', 1.)
        self.add_coupling(-J, 0, 'Sigmaz', 0, 'Sigmaz', [1,0])
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-h, u, 'Sigmax')

class DualIsingModel(CouplingMPOModel):
    def init_sites(self, model_params):
        return tenpy.networks.site.SpinHalfSite([])
    def init_terms(self, model_params):
        J = model_params.get('J', 1.)
        h = model_params.get('h', 1.)
        self.add_coupling(-J, 0, 'Sigmax', 0, 'Sigmax', [1,0])
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-h, u, 'Sigmaz')

class CritIsingModel(CouplingMPOModel):
    def init_sites(self, model_params):
        return tenpy.networks.site.SpinHalfSite([])
    def init_terms(self, model_params):
        J = model_params.get('J', 1.)
        self.add_coupling(-J, 0, 'Sigmaz', 0, 'Sigmaz', [1,0])
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-J, u, 'Sigmax')

class DualCritIsingModel(CouplingMPOModel):
    def init_sites(self, model_params):
        return tenpy.networks.site.SpinHalfSite([])
    def init_terms(self, model_params):
        J = model_params.get('J', 1.)
        self.add_coupling(-J, 0, 'Sigmax', 0, 'Sigmax', [1,0])
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-J, u, 'Sigmaz')

class XXZModel(CouplingMPOModel):
    def init_sites(self, model_params):
        return tenpy.networks.site.SpinHalfSite([])
    def init_terms(self, model_params):
        J = model_params.get('J', 1.)
        Delta = model_params.get('Delta', 0.)
        self.add_coupling(-J, 0, 'Sigmax', 0, 'Sigmax', [1,0])
        self.add_coupling(-J, 0, 'Sigmay', 0, 'Sigmay', [1,0])
        self.add_coupling(-Delta, 0, 'Sigmaz', 0, 'Sigmaz', [1,0])

class XXZModel_Sz(CouplingMPOModel):
    def init_sites(self, model_params):
        return tenpy.networks.site.SpinHalfSite(conserve='Sz')
    def init_terms(self, model_params):
        J = model_params.get('J', 1.)
        Delta = model_params.get('Delta', 0.)
        self.add_coupling(-J/2, 0, 'Sp', 0, 'Sm', [1, 0])
        self.add_coupling(-J/2, 0, 'Sm', 0, 'Sp', [1, 0])
        self.add_coupling(-Delta, 0, 'Sz', 0, 'Sz', [1, 0])

class SymmZXZModel(CouplingMPOModel):
    def init_sites(self, model_params):
        return tenpy.networks.site.SpinHalfSite([])
    def init_terms(self, model_params):
        J = model_params.get('J', 1.)
        self.add_multi_coupling(-J, [('Sigmaz', [0,0], 0), ('Sigmax', [0,1], 0), ('Sigmaz', [1,0], 0)])
        self.add_multi_coupling(-J, [('Sigmaz', [0,0], 0), ('Sigmax', [1,1], 0), ('Sigmaz', [1,0], 0)])
        self.add_multi_coupling(-J, [('Sigmaz', [0,1], 0), ('Sigmax', [0,0], 0), ('Sigmaz', [1,1], 0)])
        self.add_multi_coupling(-J, [('Sigmaz', [0,1], 0), ('Sigmax', [1,0], 0), ('Sigmaz', [1,1], 0)])

class XYModel(CouplingMPOModel):
    def init_sites(self, model_params):
        return tenpy.networks.site.SpinHalfSite([])
    def init_terms(self, model_params):
        J = model_params.get('J', 1.)
        self.add_coupling(-J, 0, 'Sigmax', 0, 'Sigmax', [1,0])
        self.add_coupling(-J, 0, 'Sigmay', 0, 'Sigmay', [1,0])

class XZModel(CouplingMPOModel):
    def init_sites(self, model_params):
        return tenpy.networks.site.SpinHalfSite([])
    def init_terms(self, model_params):
        J = model_params.get('J', 1.)
        self.add_coupling(-J, 0, 'Sigmax', 0, 'Sigmax', [1,0])
        self.add_coupling(-J, 0, 'Sigmaz', 0, 'Sigmaz', [1,0])


'''
DMRG
'''
def mydmrg1(model, systemsize, params=None, chi=300, init_state=[[1, 0],[1, 0]], bc_MPS='finite',bc1='open', max_sweeps=15, sitetype=SpinHalfSite, conserve=None, svd_min=1.e-13, max_E_err=1.e-10, to_stdout="INFO"):
    tenpy.tools.misc.setup_logging(to_stdout=to_stdout)
    L = systemsize[0]
    model_params = {
        'lattice': Lattice(systemsize,[sitetype(conserve=conserve)], order='Cstyle', bc=[bc1,'open'],bc_MPS=bc_MPS),
        'bc_MPS': bc_MPS,
        }
    if not (params is None):
        model_params.update(params)
    if bc_MPS == 'finite':
        chi_list = {0:min(10,chi), 2:min(20,chi), 4:min(50,chi), 6:min(100, chi), 8:min(200, chi), 10:min(300, chi), 12:chi}
    else:
        chi_list = {0:min(10,chi), 20:min(20,chi), 40:min(50,chi), 60:min(100, chi), 80:min(120, chi), 100:min(150, chi), 120:min(200, chi), 140:min(250, chi), 160:min(300, chi), 180:chi}
    dmrg_params = {
            'mixer': True,  # setting this to True helps to escape local minima
            'mixer_params': {
                'amplitude': 1.e-5,
#                 'decay': 1.,
                'disable_after': 5,
            },
            'trunc_params': {
                'chi_max': chi,
                'svd_min': svd_min,
                # 'trunc_cut': 0
            },
            'verbose': 1,
            'combine': True,
            'max_E_err': max_E_err,
            'chi_list':chi_list,
            'max_sweeps':max_sweeps
        }
    M = model(model_params)
    product_state = (init_state* (M.lat.N_sites))[:M.lat.N_sites]
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
    psi.canonical_form()
    eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
    E, psi = eng.run()  # equivalent to dmrg.run() up to the return parameters.
    psi.canonical_form()
    return E, psi, M

def mydmrg_excited_states(model, systemsize, params=None, chi=300, n=1,init_state=[[1, 0],[1, 0]],bc1='open', max_sweeps=15, sitetype=SpinHalfSite, conserve=None, svd_min=1.e-13, max_E_err=1.e-10, to_stdout="INFO"):
    '''
    Get excited states of *finite* DMRG.
    n: number of states to get. Note that we need E<0 to make it work.
    return: Es, states
    '''
    tenpy.tools.misc.setup_logging(to_stdout=to_stdout)
    L = systemsize[0]
    model_params = {
        'lattice': Lattice(systemsize,[sitetype(conserve=conserve)], order='Cstyle', bc=[bc1,'open'],bc_MPS='finite'),
        'bc_MPS': 'finite',
        }
    if not (params is None):
        model_params.update(params)
    M = model(model_params)

    chi_list = {0:min(10,chi), 2:min(20,chi), 4:min(50,chi), 6:min(100, chi), 8:min(200, chi), 10:min(300, chi), 12:chi}
    dmrg_params = {
            'mixer': True,  # setting this to True helps to escape local minima
            'mixer_params': {
                'amplitude': 1.e-5,
#                 'decay': 1.,
                'disable_after': 5,
            },
            'trunc_params': {
                'chi_max': chi,
                'svd_min': svd_min,
                # 'trunc_cut': 0
            },
            'verbose': 1,
            'combine': True,
            'max_E_err': max_E_err,
            'chi_list':chi_list,
            'max_sweeps':max_sweeps
        }
    product_state = (init_state* (M.lat.N_sites))[:M.lat.N_sites]
    psi0 = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
    psi0.canonical_form()
    states = dmrg_params['orthogonal_to'] = []
    Es = []
    for i in range(n):
        psi = psi0.copy()
        eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
        E, psi = eng.run() 
        states.append(psi)  # this also adds them to dmrg_params['orthogonal_to']
        Es.append(E)
    return Es, states


'''
On-site Gates
'''

def gate_onsite1(psi, gates, ind):
        assert len(ind) == len(gates)
        psit = psi.copy()
        gates_npc = [psit.sites[0].Id]*psit.L
        ind_int = (np.rint(ind)).astype(int)
        for i in range(len(ind_int)):
            gates_npc[ind_int[i]] = npc.Array.from_ndarray_trivial(np.array(gates[i]), labels=["p", "p*"])

        psit.convert_form('B')
        for i in range(psit.L):
            op = gates_npc[i]
            if isinstance(op, str):
                if op == 'Id':
                    continue  # nothing to do here...
                op = psit.sites[i].get_op(op)
            psit._B[i] = npc.tensordot(op, psit._B[i], axes=['p*', 'p'])
        # psit.canonical_form(renormalize=renormalize)

        return psit

def gate_onsite2(psi, gates, ind):
    '''
    psi: MPS
    gates: List of numpy array
    ind: list of integers, indices of sites that gates act on. Should have len(ind)=len(gates)
    '''
    assert len(ind) == len(gates)
    psit = psi.copy()
    gates_npc = [npc.Array.from_ndarray_trivial(np.array([[psit.sites[0].Id.to_ndarray()]]), labels=["wL", "wR", "p", "p*"])]*len(psi.sites)
    ind_int = (np.rint(ind)).astype(int)

    for i in range(len(ind_int)):
        gates_npc[ind_int[i]] = npc.Array.from_ndarray_trivial(np.array([[gates[i]]]), labels=["wL", "wR", "p", "p*"])

    if psi.bc == 'finite':
        mympo = MPO(psi.sites, gates_npc, bc=psi.bc, IdL=0, IdR=-1) # working with MPO is much faster than applying one-site ops one by one
    elif psi.bc == 'infinite':
        mympo = MPO(psi.sites, gates_npc, bc=psi.bc)
    else:
        raise Exception("boundary not finite or infinite.")
    mympo.apply_naively(psit)
    # psit.canonical_form()

    return psit

'''
Two-site gates
'''
def gate_twosite_nonoverlap(psi, gates, ind, cutoff): 
    '''
    psi: MPS
    gates: List of *npc* arrays, representing nearest neighbor couplings.
    ind: list of integers, left indices of sites that gates act on. Should have len(ind)=len(gates)
    '''
    assert len(ind) == len(gates)
    psit = psi.copy()
    ind_int = (np.rint(ind)).astype(int)
    for i in range(len(ind_int)):
        psit.apply_local_op(ind_int[i],gates[i],unitary=True, cutoff=cutoff)

    
    return psit

def gate_twosite2(psi, gates, ind): # Slower... Need to check!
    '''
    psi: MPS
    gates: List of *npc* arrays, representing nearest neighbor couplings.
    ind: list of integers, left indices of sites that gates act on. Should have len(ind)=len(gates)
    '''
    psit = psi.copy()
    ind_int = (np.rint(ind)).astype(int)
    H_bond = [None]*psit.L
    for i in range(len(ind_int)):
        H_bond[ind_int[i]+1] = gates[i] # check!
    M = NearestNeighborModel(Lattice([psit.L,1],[SpinHalfSite(conserve=None)], order='Cstyle', bc=['periodic','open'],bc_MPS=psit.bc), H_bond)
    mympo = M.calc_H_MPO_from_bond()
    options = {'compression_method': 'SVD',
          'trunc_params': {'chi_max':max(psit.chi), 'svd_min': 1e-10},}
    mympo.apply(psit, options)
    # psit.canonical_form()
    return psit


'''
Coefficient of MPS
'''
def a(psi:MPS, ind, initstate=[[1, 0]], flip_op=None):
    '''
    filp_op should be a numpy array.
    MPS.from_product_state works only for real coefficients if dtype not specified.
    '''
    ind_int = (np.rint(ind)).astype(int)
    zerostate = MPS.from_product_state(psi.sites, (initstate* len(psi.sites))[:len(psi.sites)],dtype=np.complex128, bc=psi.bc)
    zerostate.canonical_form()
    zerostate.norm = 1
    state0 = zerostate.copy()
    if flip_op is not None:
        for i in ind_int:
            g = npc.Array.from_ndarray_trivial(np.array(flip_op), labels=["p", "p*"])
            state0.apply_local_op(i, g)
        state0.canonical_form()
    return state0.overlap(psi)/zerostate.overlap(psi)

def V(psi:MPS, i, j, initstate=[[1, 0]], flip_op=None):
    return a(psi, [i,j],initstate, flip_op) - a(psi, [i],initstate, flip_op) * a(psi, [j],initstate, flip_op)

'''
Fit and plot
'''
def fitplot(ax:Axes, x, y, fitfunc=None, fitrange=None, initial_guess=None, xscale='linear', yscale='linear', dataplotstyle='o-', fitplotstyle='k--',label='data', **kwargs):
    assert len(x) == len(y)
    ax.plot(x,y,dataplotstyle, label=label)
    params = None
    if fitfunc is not None:
        if fitrange is not None:
            fitrange = (np.rint(fitrange)).astype(int)
        else:
            fitrange = range(len(x))
        xfit = x[fitrange]
        yfit = y[fitrange]
        params, _ = scipy.optimize.curve_fit(fitfunc, xfit, yfit, p0=initial_guess, **kwargs)
        y_pred = fitfunc(xfit, *params)
        ax.plot(xfit,y_pred,fitplotstyle,label=str(params))

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    return params

'''
Correlation functions
'''
def correlation(psi:MPS, op1, op2=None, ind=[0], connected=True):
    '''
    compute <op1_j op1_k>, for op1 != op2 cases, use tenpy functions.
    op1 and op2 should be either both ndarray or both list or both TermList
    return corr(ind[0], ind[i]) for i in ind. The first entry should be 1 if op^2 = Id.
    '''
    ind_int = (np.rint(ind)).astype(int)
    if isinstance(op1, np.ndarray):
        op1 = npc.Array.from_ndarray_trivial(np.array(op1), labels=["p", "p*"])
        if op2 is not None:
            op2 = npc.Array.from_ndarray_trivial(np.array(op2), labels=["p", "p*"])
        else:
            op2 = op1
        corr = psi.correlation_function(op1, op2, [ind_int[0]], ind_int[1:])[0]
        if connected:
            ev0 = psi.expectation_value(op1, ind_int[0])
            ev = ev0 * psi.expectation_value(op2, ind_int[1:]) # expectation values
            return corr - ev
        else:
            return corr
    elif isinstance(op1, list):
        if op2 is None:
            op2 = op1
        corr = psi.term_correlation_function_right(op1, op2, ind_int[0], ind_int[1:])
        if connected:
            op1_tuple = [(i[0],i[1]+ind_int[0]) for i in op1]
            ev0 = psi.expectation_value_term(op1_tuple)
            ev = ev0 * np.array([psi.expectation_value_term([(i[0],i[1]+k) for i in op2]) for k in ind_int[1:]])
            return corr - ev
        else:
            return corr
    elif isinstance(op1, TermList):
        if op2 is None:
            op2 = op1
        corr = psi.term_list_correlation_function_right(op1, op2, ind_int[0], ind_int[1:])
        if connected:
            ev0 = sum([psi.expectation_value_term(term)*strength for term, strength in op1.shift(ind_int[0])])
            ev = ev0 * np.array([sum([psi.expectation_value_term(term)*strength for term, strength in op2.shift(k)]) for k in ind_int[1:]])
            return corr - ev
        else:
            return corr
    else:
        raise Exception("invalid operator type.")

'''
Misc
'''
def setlog(to_stdout, filename=None):
    tenpy.tools.misc.setup_logging(to_stdout=to_stdout,filename=filename)

def compress(psi, chi):
    '''
    compress an MPS to bond dimension chi
    '''
    pass

def scale(x):
    '''
    x: numpy array
    return: x scaled to [0,1]
    '''
    xmin = np.min(x)
    xmax = np.max(x)
    y = (x-xmin)/(xmax-xmin)
    return y

def chord(x,L):
    '''
    chord distance
    '''
    return L/np.pi*np.sin(np.pi*x/L)

def loopdistance(n, a, b):
    '''
    distance in periodic chain
    '''
    if (a > b):
        a, b = b, a
    clock_wise_dist = b - a
    counter_clock_wise_dist = (a - 1) + (n - b + 1)
    minimum_dist = min(clock_wise_dist, counter_clock_wise_dist)
    return minimum_dist

# fit central charge
def fit_central_charge(EE, Nboundary=1, fitrange=None, ax=None, label=None):
    x = np.arange(len(EE)) + 1
    x_chord = chord(x, len(EE)+1)
    if fitrange is not None:
        fitrange = (np.rint(fitrange)).astype(int)
        x_chord_ = x_chord[fitrange]
        EE_ = EE[fitrange]
        x_ = x[fitrange]
    else:
        x_chord_ = x_chord
        EE_ = EE
        x_ = x
    k, b = np.polyfit(np.log(x_chord_), EE_, 1)
    c = k*6/Nboundary
    if ax is not None:
        if label is not None:
            l = label + ', c='+str(round(c,3))
        else:
            l = 'c='+str(round(c,3))
        ax.plot(x_ - 1, k*np.log(x_chord_)+b, 'ko--', label=l)
    return c

# fit central charge
def fit_central_charge_nolabel(EE, Nboundary=1, fitrange=None, ax=None, label=None):
    x = np.arange(len(EE)) + 1
    x_chord = chord(x, len(EE)+1)
    if fitrange is not None:
        fitrange = (np.rint(fitrange)).astype(int)
        x_chord_ = x_chord[fitrange]
        EE_ = EE[fitrange]
        x_ = x[fitrange]
    else:
        x_chord_ = x_chord
        EE_ = EE
        x_ = x
    k, b = np.polyfit(np.log(x_chord_), EE_, 1)
    c = k*6/Nboundary
    if ax is not None:
        # if label is not None:
        #     l = label + ', c='+str(round(c,3))
        # else:
        #     l = 'c='+str(round(c,3))
        ax.plot(x_ - 1, k*np.log(x_chord_)+b, 'ko--')
    return c

'''
bit array
'''
def bit2Int(qubit):
    qubit = np.rint(qubit).astype(int)
    return int(''.join(str(i) for i in np.rint(qubit).astype(int)),2)
def Int2bit(n,L):
    return np.array([int(x) for x in bin(n)[2:].zfill(L)])
def spinHalf2Int(spinHalf):
    spinHalf = np.rint(spinHalf).astype(int)
    return bit2Int((spinHalf+1)/2)
def Int2spinHalf(n,L):
    y = 2*Int2bit(n,L)-1
    return np.rint(y).astype(int)


'''
Fast Sampling from MPS
'''
def sample_projective_measurements(psi, first_site=0, last_site=None, ops=None, rng=None, norm_tol=1e-12, complex_amplitude=True):
    '''
    A wrapper or Tenpy's original sample_measurements function.
    Returns sigmas, weight
    For details, see https://github.com/tenpy/tenpy/blob/v1.0.6/tenpy/networks/mps.py#L3837-L3934
    '''
    return psi.sample_measurements(first_site=first_site, last_site=last_site, ops=ops, rng=rng, norm_tol=norm_tol, complex_amplitude=complex_amplitude)


class POVM():
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


def sample_povm_measurements(psi, first_site=0, last_site=None, ops=None, rng=None, norm_tol=1e-12, complex_amplitude=True):
    '''
    Sample POVM measurement outcomes using a similar algorithm as Tenpy's sample_measurements function.
    '''
    
    # TODO: Check!
    
    if tuple(psi._p_label) != ('p', ):
        raise NotImplementedError("Only works for a single physical 'p' leg")
    if last_site is None:
        last_site = psi.L - 1
    if rng is None:
        rng = np.random.default_rng()
    if not ops:
        raise ValueError("ops must be a non-empty list of POVM objects.")
    
    povm_kraus = []
    for povm in ops:
        if not isinstance(povm, POVM):
            raise TypeError("ops must be a list of POVM instances.")
        if len(povm.kraus_ops) != len(povm.outcomes):
            raise ValueError("Each POVM must have the same number of kraus_ops and outcomes.")
        kraus_list = []
        for K in povm.kraus_ops:
            kraus_list.append(npc.Array.from_ndarray_trivial(K, labels=['p_out', 'p_in']))
        povm_kraus.append(kraus_list)

    sigmas = []
    total_weight = 1.
    theta = psi.get_theta(first_site, n=1).replace_label('p0', 'p')
    for i in range(first_site, last_site + 1):
        # theta = wave function in basis vL [sigmas...] p vR
        # where the `sigmas` are already fixed to the measurement results
        povm_idx = (i - first_site) % len(povm_kraus)
        current_kraus = povm_kraus[povm_idx]
        current_outcomes = ops[povm_idx].outcomes
        probs = []
        theta_candidates = []
        for K in current_kraus:
            theta_tmp = npc.tensordot(theta, K, axes=[['p'], ['p_in']])
            theta_tmp = theta_tmp.transpose([0, 2, 1]).replace_label('p_out', 'p')
            prob = npc.norm(theta_tmp)**2
            theta_candidates.append(theta_tmp)
            probs.append(prob)
        prob_sum = float(np.sum(probs).real)
        if abs(prob_sum - 1.0) > norm_tol:
            raise ValueError("Probability sum not equal to 1. Tolerance exceeded.")
        probs = np.asarray(probs, dtype=float) / prob_sum
        sigma_idx = int(rng.choice(len(probs), p=probs))
        sigmas.append(current_outcomes[sigma_idx])
        theta = theta_candidates[sigma_idx]
        weight = npc.norm(theta)
        total_weight *= weight
        if i != last_site:
            theta = theta / weight
            theta_mat = theta.combine_legs([['vL', 'p']], new_axes=[0])
            theta_mat = theta_mat.replace_label(theta_mat.get_leg_labels()[0], 'vL')
            theta_mat = theta_mat.replace_label(theta_mat.get_leg_labels()[1], 'vR')
            _, R = npc.qr(theta_mat, inner_labels=['vR', 'vL'])
            B = psi.get_B(i + 1)
            theta = npc.tensordot(R, B, axes=['vR', 'vL'])
        elif psi.bc == 'finite' and first_site == 0 and last_site == psi.L - 1:
            theta_scalar = theta.to_ndarray().reshape(-1).sum()
            total_weight = total_weight * theta_scalar / weight
    if not complex_amplitude:
        total_weight = np.abs(total_weight)**2
    return sigmas, total_weight

def weak_measurement_pauli(op : np.ndarray, beta, real=False) -> POVM:
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

'''
Finish.
'''
print("Imported MPSToolBox")


'''
Temperary testing area
'''
psi = load_pkl("notebooks/TCI_L100_chi500_PBC.pkl")

rng = np.random.default_rng(35)
s, weight = sample_projective_measurements(psi, first_site=0, ops=['Sigmax'], rng=rng)
print([int(i) for i in s])
print(weight)

rng = np.random.default_rng(35)
povm = POVM(kraus_ops=[(Id - sX)/2, (Id + sX)/2], outcomes=[-1,+1])
s, weight = sample_povm_measurements(psi, first_site=0 ,ops=[povm], rng=rng)
print(s)
print(weight)

rng = np.random.default_rng(35)
povm = weak_measurement_pauli(sX, beta=2, real=True)

s, weight = sample_povm_measurements(psi, first_site=0 ,ops=[povm], rng=rng)
print(s)
print(weight)

