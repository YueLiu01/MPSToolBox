import MPSToolBox as mtb
import numpy as np
psi = mtb.load_pkl("../wavefunctions/CritIsingModel_L100_chi300_PBC_.pkl")


beta = 0.5
povm = mtb.weak_measurement_pauli(mtb.sZ, beta=2, real=True)
for seed in np.arange(10):
    rng = np.random.default_rng(seed=seed)
    s, weight = mtb.sample_povm_measurements(psi, first_site=0, ops=[povm], rng=rng)
    print(s)
    gates = [mtb.expm(beta * s[i] * mtb.sZ) for i in range(psi.L)]
    psi_measured = mtb.gate_onsite1(psi, gates, np.arange(psi.L))
    psi_measured.canonical_form()
    s2, weight2 = mtb.sample_projective_measurements(psi_measured, first_site=0, ops=['Sigmaz'], rng=rng)
    s2 = [round(s2[i]) for i in range(len(s2))]
    print(s2)
    
    outcome = {'ancilla': s, 'system': s2}
    print(outcome)
    print('---')
    
    
    
    
# '''
# Temperary testing area
# '''
# psi = load_pkl("notebooks/TCI_L100_chi500_PBC.pkl")

# rng = np.random.default_rng(35)
# s, weight = sample_projective_measurements(psi, first_site=0, ops=['Sigmax'], rng=rng)
# print([int(i) for i in s])
# print(weight)

# rng = np.random.default_rng(35)
# povm = POVM(kraus_ops=[(Id - sX)/2, (Id + sX)/2], outcomes=[-1,+1])
# s, weight = sample_povm_measurements(psi, first_site=0 ,ops=[povm], rng=rng)
# print(s)
# print(weight)

# rng = np.random.default_rng(35)
# povm = weak_measurement_pauli(sX, beta=2, real=True)
# s, weight = sample_povm_measurements(psi, first_site=0 ,ops=[povm], rng=rng)
# print(s)
# print(weight)
