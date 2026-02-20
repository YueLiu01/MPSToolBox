import numpy as np
import MPSToolBox as my
import matplotlib.pyplot as plt
np.set_printoptions(precision=5, suppress=True, linewidth=100)


N = 10000

# _connected_corr = Zcorr - Z[:, :1] * Z
L = 200
beta = 0.5
# load <Z_0 Z_r>_s, where each row corresponds to a different snapshot and each column corresponds to a different distance r.
Zcorr = np.load(f"../data/Fast_MPS_CritIsing_L{L}_beta{beta}_Zcorr_N{N}_0_r.npy")
# load <Z_r>_s, where each row corresponds to a different snapshot and each column corresponds to a different position r.
Z = np.load(f"../data/Fast_MPS_CritIsing_L{L}_beta{beta}_Zexp_N{N}_0_r.npy")

def A(q, Zcorr=Zcorr, Z=Z):
    return np.mean(np.power(Zcorr, q), axis=0) # A(q;r) := <Z_0 Z_r>^q, averaged over snapshots

def A_abs(q, Zcorr=Zcorr, Z=Z):
    return np.mean(np.power(np.abs(Zcorr), q), axis=0) # A_abs(q;r) := <|Z_0 Z_r|^q>, averaged over snapshots

def A_c(q, Zcorr=Zcorr, Z=Z):
    return np.abs(np.mean(np.power(Zcorr, q) - Z[:, :1]**q * Z**q, axis=0)) # A_c(q;r) := |<Z_0 Z_r>^q - <Z_0>^q <Z_r>^q|, averaged over snapshots

def B(q, Zcorr=Zcorr, Z=Z):
    _connected_corr = Zcorr - Z[:, :1] * Z
    return np.mean(np.power(_connected_corr, q), axis=0) # B(q;r) := <(Z_0 Z_r - <Z_0><Z_r>)^q>, averaged over snapshots.

def B_abs(q, Zcorr=Zcorr, Z=Z):
    _connected_corr = Zcorr - Z[:, :1] * Z
    return np.mean(np.power(np.abs(_connected_corr), q), axis=0) # B_abs(q;r) := <|Z_0 Z_r - <Z_0><Z_r>|^q>, averaged over snapshots.

def plot_averaged_moments(Zcorr, Z, q_list, func, ylabel, title_suffix=""):
    r = np.arange(Zcorr.shape[1])
    plt.figure()
    for q in q_list:
        plt.plot(r, func(q, Zcorr=Zcorr, Z=Z), label=f"{ylabel}({q})")
    plt.xlabel("r")
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(f"{ylabel}(q) vs r for L={L}, beta={beta}" + title_suffix)
    plt.tight_layout()
    plt.savefig(f"temp/{ylabel}_L{L}_beta{beta}_N{N}.png")
    plt.close()
    
def plot_averaged_moments(Zcorr, Z, q_list, func, ylabel, title_suffix="", loglog=False, fit=False, fit_range=(1, 5), fit_const=False):
    r = np.arange(Zcorr.shape[1])
    if loglog:
        r_chord = (L / np.pi) * np.sin(np.pi * r / L) # chord distance
        r = r_chord
    plt.figure()
    for q in q_list:
        # plt.plot(r, func(q, Zcorr=Zcorr, Z=Z), label=f"{ylabel}({q})")
        plt.plot(r, func(q, Zcorr=Zcorr, Z=Z), label=f"{ylabel}(q={q})")
        const = 0
        if fit_const:
            def fit_func(x, a, b, c):
                x_chord = (L / np.pi) * np.sin(np.pi * x / L)
                return a * x_chord**(-b) + c
            from scipy.optimize import curve_fit
            popt, _ = curve_fit(fit_func, r[1:-1], func(q, Zcorr=Zcorr, Z=Z)[1:-1])
            fitted_line = fit_func(r, *popt)
            plt.plot(r, fitted_line, 'k--', label=f"Fit for {ylabel}({q}), a={popt[0]:.2e}, b={popt[1]:.2f}, c={popt[2]:.2e}")
            const = popt[2]
        if fit and loglog:
            # Fit a power law to the data in the specified range
            fit_mask = (r >= fit_range[0]) & (r <= fit_range[1])
            log_r_fit = np.log(r[fit_mask])
            log_func_fit = np.log(func(q, Zcorr=Zcorr, Z=Z)[fit_mask])
            coeffs = np.polyfit(log_r_fit, log_func_fit, 1)
            fitted_line = np.exp(coeffs[1]) * r**coeffs[0]
            plt.plot(r, fitted_line, 'k--', label=f"Fit for {ylabel}({q}), slope={coeffs[0]:.2f}")
    plt.xlabel("r")
    if loglog:
        plt.xlabel("Chord distance r'")
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(f"{ylabel}(q) vs r for L={L}, beta={beta}" + title_suffix)
    if loglog:
        plt.xscale("log")
        plt.yscale("log")
    plt.tight_layout()
    save_suffix = "_loglog" if loglog else ""
    plt.savefig(f"temp/{ylabel}_L{L}_beta{beta}_N{N}{save_suffix}.png")
    plt.close()
    
def plot_averaged_moments_vs_beta(L, beta_list, q_list, func, ylabel, title_suffix=""):
    for q in q_list:
        plt.figure()
        for beta in beta_list:
            Zcorr = np.load(f"../data/Fast_MPS_CritIsing_L{L}_beta{beta}_Zcorr_N{N}_0_r.npy")
            Z = np.load(f"../data/Fast_MPS_CritIsing_L{L}_beta{beta}_Zexp_N{N}_0_r.npy")
            r = np.arange(Zcorr.shape[1])
            plt.plot(r, func(q, Zcorr=Zcorr, Z=Z), label=f"{ylabel}({q}), beta={beta}")
        plt.xlabel("r")
        plt.ylabel(ylabel)
        plt.legend()
        plt.title(f"{ylabel}(q) vs r for L={L}, N={N}" + title_suffix)
        plt.tight_layout()
        plt.savefig(f"temp/{ylabel}_L{L}_N{N}_q{q}.png")
        plt.close()
    
# plot_averaged_moments_vs_beta(20, beta_list=[0.1, 0.2, 0.3, 0.5, 0.8, 1.0], q_list=[1, 2, 3], func=A, ylabel="A", title_suffix=r", $\overline{\langle Z_0 Z_r \rangle_s^q}$")

# plot_averaged_moments_vs_beta(20, beta_list=[0.1, 0.2, 0.3, 0.5, 0.8, 1.0], q_list=[0.1, 0.5, 1, 1.5, 2], func=A_abs, ylabel="A_abs", title_suffix=r", $\overline{|\langle Z_0 Z_r \rangle_s|^q}$")

# plot_averaged_moments_vs_beta(20, beta_list=[0.1, 0.2, 0.3, 0.5, 0.8, 1.0], q_list=[1, 2, 3], func=B, ylabel="B", title_suffix=r", $\overline{(\langle Z_0 Z_r \rangle_s - \langle Z_0 \rangle_s \langle Z_r \rangle_s)^q}$")



L = 200
beta = 0.1
# load <Z_0 Z_r>_s, where each row corresponds to a different snapshot and each column corresponds to a different distance r.
Zcorr = np.load(f"../data/Fast_MPS_CritIsing_L{L}_beta{beta}_Zcorr_N{N}_0_r.npy")
# load <Z_r>_s, where each row corresponds to a different snapshot and each column corresponds to a different position r.
Z = np.load(f"../data/Fast_MPS_CritIsing_L{L}_beta{beta}_Zexp_N{N}_0_r.npy")
q_list = [1, 2, 3, 4, 5, 6]
plot_averaged_moments(Zcorr, Z, q_list, A_abs, "A_abs", title_suffix=r", $\overline{|\langle Z_0 Z_r \rangle_s|^q}$", fit_const=True)
# plot_averaged_moments(Zcorr, Z, q_list, A_c, "A_c", title_suffix=r", $\overline{|\langle Z_0 Z_r \rangle_s^q - \langle Z_0 \rangle_s^q \langle Z_r \rangle_s^q|}$", fit=True, fit_range=(10, 50), loglog=True)

# L = 100
# beta = 1.0
# # load <Z_0 Z_r>_s, where each row corresponds to a different snapshot and each column corresponds to a different distance r.
# Zcorr = np.load(f"../data/Fast_MPS_CritIsing_L{L}_beta{beta}_Zcorr_N{N}_0_r.npy")
# # load <Z_r>_s, where each row corresponds to a different snapshot and each column corresponds to a different position r.
# Z = np.load(f"../data/Fast_MPS_CritIsing_L{L}_beta{beta}_Zexp_N{N}_0_r.npy")

# # plot_averaged_moments(Zcorr, Z, q_list, A_abs, "A_abs", title_suffix=r", $\overline{|\langle Z_0 Z_r \rangle_s|^q}$")
# plot_averaged_moments(Zcorr, Z, q_list, B, "B", title_suffix=r", $\overline{(\langle Z_0 Z_r \rangle_s - \langle Z_0 \rangle_s \langle Z_r \rangle_s)^q}$", loglog=True, fit=True, fit_range=(1, 10))
# # plot_averaged_moments(Zcorr, Z, q_list, B_abs, "B_abs", title_suffix=r", $\overline{|\langle Z_0 Z_r \rangle_s - \langle Z_0 \rangle_s \langle Z_r \rangle_s|^q}$")
