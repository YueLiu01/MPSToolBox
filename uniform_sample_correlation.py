import os
from multiprocessing import Pool, cpu_count

# Avoid thread oversubscription inside workers
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import MPSToolBox as my

np.set_printoptions(precision=5, suppress=True, linewidth=100)

_WORKER_PSI = None
_WORKER_L = None
_WORKER_BETA = None
_WORKER_SITE_INDICES = None
_WORKER_GATE_CACHE = None


def _init_worker(psi_path, L, beta):
    """Initialize process-local data once per worker."""
    global _WORKER_PSI, _WORKER_L, _WORKER_BETA, _WORKER_SITE_INDICES, _WORKER_GATE_CACHE
    _WORKER_PSI = my.load_pkl(psi_path)
    _WORKER_L = int(L)
    _WORKER_BETA = float(beta)
    _WORKER_SITE_INDICES = np.arange(_WORKER_L)
    _WORKER_GATE_CACHE = {}


def _cached_gate(outcome_value):
    gate = _WORKER_GATE_CACHE.get(outcome_value)
    if gate is None:
        gate = my.expm(_WORKER_BETA * outcome_value * my.sZ)
        _WORKER_GATE_CACHE[outcome_value] = gate
    return gate


def _process_outcome(s):
    gates = [_cached_gate(s[i]) for i in range(_WORKER_L)]
    psi_measured = my.gate_onsite1(_WORKER_PSI, gates, _WORKER_SITE_INDICES)
    try:
        psi_measured.canonical_form()
        overlap = _WORKER_PSI.overlap(psi_measured)
        pr = float(np.abs(overlap)**2)
        z_exp = psi_measured.expectation_value("Sigmaz")
        z_corr = psi_measured.correlation_function(
            "Sigmaz", "Sigmaz", [0], _WORKER_SITE_INDICES
        )[0]
        return z_corr, z_exp, pr
    finally:
        # Ensure the large temporary MPS is released as soon as this task ends.
        del psi_measured


def _run_single_case(L, beta, N, n_workers=None):
    anc_data = np.random.choice([-1, 1], size=(N, L))
    np.save(f"temp/Uniform_L{L}_beta{beta}_N{N}.npy", anc_data)
    outcomes = anc_data[:N]
    if outcomes.shape[0] == 0:
        return

    if n_workers is None:
        n_workers = min(cpu_count(), outcomes.shape[0])
    else:
        n_workers = min(max(1, int(n_workers)), outcomes.shape[0])

    psi_path = f"../wavefunctions/CritIsingModel_L{L}_chi300_PBC_.pkl"
    chunksize = max(1, outcomes.shape[0] // (n_workers * 8))

    max_tasks_per_child = int(os.getenv("MAX_TASKS_PER_CHILD", "200"))

    z_corr_list = []
    z_exp_list = []
    pr_list = []

    with Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(psi_path, L, beta),
        maxtasksperchild=max_tasks_per_child,
    ) as pool:
        for z_corr, z_exp, pr in pool.imap(_process_outcome, outcomes, chunksize=chunksize):
            z_corr_list.append(z_corr)
            z_exp_list.append(z_exp)
            pr_list.append(pr)

    z_corr_arr = np.array(z_corr_list)
    z_exp_arr = np.array(z_exp_list)
    np.save(f"temp/Fast_MPS_CritIsing_L{L}_beta{beta}_Zcorr_uniform_N{N}_0_r.npy", z_corr_arr)
    np.save(f"temp/Fast_MPS_CritIsing_L{L}_beta{beta}_Zexp_uniform_N{N}_0_r.npy", z_exp_arr)
    np.save(f"temp/Fast_MPS_CritIsing_L{L}_beta{beta}_Pr_uniform_N{N}_0_r.npy", np.array(pr_list))

def _parse_list_env(name, cast, default):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    values = []
    for token in raw.split(","):
        token = token.strip()
        if token == "":
            continue
        values.append(cast(token))
    if not values:
        raise ValueError(f"{name} is set but contains no valid values.")
    return values


def main():
    N = int(os.getenv("N", "10000"))
    L_list = _parse_list_env("L_LIST", int, [20])
    beta_list = _parse_list_env("BETA_LIST", float, [0.5])
    env_workers = os.getenv("N_WORKERS")
    n_workers = int(env_workers) if env_workers is not None else None

    for L in L_list:
        for beta in beta_list:
            print(L, beta, flush=True)
            _run_single_case(L, beta, N, n_workers=n_workers)


if __name__ == "__main__":
    main()
