import sys
from pathlib import Path
import pickle
import numpy as np

parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from toy_models import L96, L96_adm
from solve_ode import runge_kuta4, rk4_nl_adm
from DA_utilis_fixed import h, Dh, FourDVar_practical

# ============================================================
# Configuration
# ============================================================
K = 40
F = 8.0
model_dt = 0.05
DA_dt = 0.005
DA_window = 0.2
nmember = 50
ncase = 200
seed = 42

# Covariances actually used to GENERATE errors and in 4D-Var.
bg_coeff = 0.5
obs_coeff = 0.3

# Debug switch: True => control observation is exact truth.
# EDA members are still y_control + N(0, R_da).
PERFECT_CONTROL_OBS = True

with open('database.pkl', 'rb') as f:
    data = pickle.load(f)
real = data['real']
B_base = data['B']

# Keep generation covariance and DA covariance identical.
B_da = bg_coeff * B_base
R_da = np.diag(obs_coeff * np.diag(B_base))

_, nstep = real.shape
forecast_idx = np.arange(0, nstep, 100)[:ncase]

# Model points are t=0, dt, 2dt, ... .
# Sparse observations are at steps 1,3,5,... inside the DA window.
N_model = int(round(DA_window / DA_dt))
obs_idx = np.arange(1, N_model, 2, dtype=int)
nobs = len(obs_idx)

rng = np.random.default_rng(seed)

initial_state = np.zeros((K, nmember + 1, len(forecast_idx)))
observation = np.zeros((K, nobs, nmember + 1, len(forecast_idx)))
truth_initial = np.zeros((K, len(forecast_idx)))

l96 = lambda x: L96(x, F)
l96_adm = lambda a, x: L96_adm(a, x)
rk4 = lambda fun, x: runge_kuta4(fun, x, DA_dt)
rk4_adm = lambda adm, nlm, zt, lam: rk4_nl_adm(adm, nlm, zt, lam, DA_dt)
DA = FourDVar_practical(l96, l96_adm, rk4, rk4_adm)


def build_database():
    """Build common-background / perturbed-observation EDA inputs."""
    for icase, start_idx in enumerate(forecast_idx):
        truth0 = real[:, start_idx].copy()
        truth_initial[:, icase] = truth0

        # One common background for control + all EDA members.
        bg_error = rng.multivariate_normal(np.zeros(K), B_da)
        xb = truth0 + bg_error
        initial_state[:, :, icase] = xb[:, None]

        # Truth trajectory on exactly the same RK4 + DA_dt used by 4D-Var.
        truth_traj = np.zeros((K, N_model))
        truth_traj[:, 0] = truth0
        for istep in range(N_model - 1):
            truth_traj[:, istep + 1] = runge_kuta4(l96, truth_traj[:, istep], DA_dt)

        truth_obs = truth_traj[:, obs_idx]

        # Control observation.
        if PERFECT_CONTROL_OBS:
            y_control = truth_obs.copy()
        else:
            e0 = rng.multivariate_normal(np.zeros(K), R_da, size=nobs).T
            y_control = truth_obs + e0
        observation[:, :, 0, icase] = y_control

        # EDA members: same observed dataset + independent observation perturbations.
        e = rng.multivariate_normal(
            np.zeros(K), R_da, size=(nmember, nobs)
        ).transpose(2, 1, 0)  # (K, nobs, nmember)
        observation[:, :, 1:, icase] = y_control[:, :, None] + e


def truth_obs_sanity(icase):
    """Exact truth must reproduce perfect observations at obs_idx."""
    truth0 = truth_initial[:, icase]
    traj = np.zeros((K, N_model))
    traj[:, 0] = truth0
    for i in range(N_model - 1):
        traj[:, i + 1] = rk4(l96, traj[:, i])

    y0 = observation[:, :, 0, icase]
    max_innov = 0.0
    if PERFECT_CONTROL_OBS:
        for j, step in enumerate(obs_idx):
            max_innov = max(max_innov, np.linalg.norm(y0[:, j] - h(traj[:, step], K)))
    return max_innov


def run_control_4dvar(icase=0, verbose=True):
    xb = initial_state[:, 0, icase]
    truth0 = truth_initial[:, icase]
    y = observation[:, :, 0, icase]

    xa, result = DA.four_dims_var_optimizer_scipy(
        xb, B_da, y, R_da, obs_idx, N_model, Dh, h,
        max_iter=200, tol=1e-7, verbose=verbose, return_result=True
    )

    stats = {
        'bg_error': np.linalg.norm(xb - truth0),
        'ana_error': np.linalg.norm(xa - truth0),
        'increment': np.linalg.norm(xa - xb),
        'cost_final': result.fun,
        'nit': result.nit,
        'success': result.success,
    }
    return xa, stats


def run_eda(icase=0, members=None):
    """Run an ensemble of independent 4D-Var analyses."""
    if members is None:
        members = range(1, nmember + 1)

    xb = initial_state[:, 0, icase]
    truth0 = truth_initial[:, icase]
    analyses = []

    for imem in members:
        ymem = observation[:, :, imem, icase]
        xa = DA.four_dims_var_optimizer_scipy(
            xb, B_da, ymem, R_da, obs_idx, N_model, Dh, h,
            max_iter=150, tol=1e-6, verbose=False
        )
        analyses.append(xa)

    analyses = np.stack(analyses, axis=1)
    mean_a = analyses.mean(axis=1)
    spread = np.sqrt(np.mean(np.var(analyses, axis=1, ddof=1))) if analyses.shape[1] > 1 else 0.0

    return analyses, {
        'bg_error': np.linalg.norm(xb - truth0),
        'eda_mean_error': np.linalg.norm(mean_a - truth0),
        'mean_member_error': np.mean(np.linalg.norm(analyses - truth0[:, None], axis=0)),
        'spread_rms': spread,
    }


if __name__ == '__main__':
    build_database()

    case = 100
    print('obs_idx =', obs_idx)
    print('max perfect-truth innovation =', truth_obs_sanity(case))

    _, stats = run_control_4dvar(case, verbose=True)
    print('CONTROL:', stats)

    # Fast smoke test of EDA; change to members=None for all 50 members.
    _, estats = run_eda(case, members=range(1, 6))
    print('EDA (first 5 members):', estats)
