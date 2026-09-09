from baro_utilis import BARO_VORT, ift, ft
import numpy as np
import pickle,json
from math import sqrt
## set model parameters
with open("config.json", "r") as f:
    config = json.load(f)

dt = config['model']['dt']
nx = config['model']['nx']
ny = config['model']['ny']

model = BARO_VORT(config['model'])

## initial state
z, zt = model.initial_state_jet()
## apply fixed forcing in spectral space by exciting certain wavenumbers
forcet = model.forcing_operator(z.shape)
## spin up
spinup_tu = config['spinup_and_control']['spinup_tu']
steps = int(spinup_tu/dt)
for i in range(steps):
    zt = model.bve_propagator(zt,forcet,verbose=True)
## sampling to calculate matrix
# dimensions of spectra space
lx, ly = zt.shape
# number of sample
K = config['spinup_and_control']['approx_saved_states']
# interval of sample
sample_tu = config['spinup_and_control']['save_interval_tu']
# tu of observation.
da_window = config['data_assimilation']['four_d_var_window_tu']
da_tu = config['data_assimilation']['cycle_interval_tu']
# case of forecast
nfcst   = config['forecast_evaluation']['n_cases']
fcst_tu = config['forecast_evaluation']['forecast_length_tu']
ver_tu  = config['forecast_evaluation']['verification_interval_tu']
# initial error
initial_coeff = config['background']['initial_coeff']
# obs error
obs_coeff = config['observations']["R"]['obs_coeff']
# interval of observation
nobs = config['observations']['nobs']
obs_interval = (nx*ny)/nobs
obs_interval_real = int(nx/sqrt(nobs))

## calculate climatological matrix
zbase = np.zeros((K,lx,ly),dtype=np.complex128)
zphy  = np.zeros((K,nx,ny),dtype=np.float32)
k = 0
while k < K:
    for i in range(int(sample_tu/dt)):
        zt = model.bve_propagator(zt,forcet,verbose=True)

    zphy[k,:,:] = ift(zt)
    zbase[k,:,:] = zt
    k += 1

## store it in spectra domain   
#C = np.cov(zphy.reshape(K, -1).T)
real_anom = zphy - zphy.mean(axis=0,keepdims=True)
diag_mean = np.mean(np.sum(real_anom**2, axis=0))/(K-1)
frac = config['background']['regularization_if_full_B']['fraction']
#Breg = B + frac*np.diag(B).mean()*np.eye(nx*ny)
lamc = frac*diag_mean

## calculate background error matrix
zphyB = zphy[:K//2].copy()
zbaseB= zbase[:K//2]
zphyBp= zphyB.copy()
# disturb in spectra domain.
mean_spec = np.mean(zbase, axis=0)          
anom_spec = zbase - mean_spec               
var_spec = np.var(anom_spec, axis=0, ddof=1)     
std_spec = np.sqrt(var_spec)                      

rng = np.random.default_rng(seed=42)
# gaussian noise.
phys_noise = rng.normal(0, 1, size=(K//2, nx, ny))
spec_noise = (np.fft.fft2(phys_noise, axes=(-2, -1))/ np.sqrt(nx * ny)) 
# perturbation in spectral domain
spectral_perturbation = initial_coeff* spec_noise * std_spec
zbaseBp = zbaseB + spectral_perturbation
#z_error = rng.multivariate_normal(mean=np.zeros(C.shape[0]), cov=initial_coeff * C, size=(K//2))

for isample in range(K//2):
    ztp = zbaseBp[isample,:,:]
    zt  = zbaseB[isample,:,:]
    for i in range(int(da_tu/dt)):
        ztp = model.bve_propagator(ztp,forcet,verbose=True)
        zt  = model.bve_propagator(zt,forcet,verbose=True)
    
    zphyBp[isample] = ift(ztp)
    zphyB[isample]  = ift(zt)
# change to physical space
error = zphyBp.reshape(K//2,-1) - zphyB.reshape(K//2,-1)
error_center = error - error.mean(axis=0,keepdims=True)
#B = np.cov(error)
# regularize
diag_mean = np.mean(np.sum(error_center**2, axis=0))/(K//2-1)
frac = config['background']['regularization_if_full_B']['fraction']
#Breg = B + frac*np.diag(B).mean()*np.eye(nx*ny)
lam = frac*diag_mean

## generate forecast truth
# obtain state at the start of assimilation.
z_evolve  = np.zeros((nfcst,int(da_window/da_tu),nx,ny),dtype=np.float32)
z_initial = np.zeros((nfcst,nx,ny),dtype=np.float32)
k = 0
zt = zbase[-1,:,:]
while k < nfcst:
    # initial state
    for i in range(int(2*int(da_window/dt))):
        zt = model.bve_propagator(zt,forcet,verbose=True)    
    z_initial[k] = ift(zt)
    # evolve state
    t = 0.; i = 0
    for j in range(int(da_window/dt)):
        t = (j + 1) * dt
        zt = model.bve_propagator(zt,forcet,verbose=True)
        if (j + 1) % int(round(da_tu/dt)) == 0:
            z_evolve[k,i] = ift(zt)
            i += 1
    k += 1

## generate observation. (add R to z_evolve)
zphy_anom = zphy - np.mean(zphy,axis=0)
diag_C = np.sum(zphy_anom.reshape(K,-1)**2,axis=0) / (K-1)

obs_idx_2d = np.arange(nx*ny).reshape(nx, ny)[::obs_interval_real,::obs_interval_real]
obs_idx = obs_idx_2d.ravel()

obs_var = (obs_coeff**2) * diag_C[obs_idx]
R = np.diag(obs_var)

std_obs = np.sqrt(obs_var)
obs_error_flat = rng.normal(0., std_obs, size=(nfcst, int(da_window/da_tu), len(std_obs)))
obs_error_field = np.zeros((nfcst, int(da_window/da_tu), nx, ny), dtype=np.float64)
obs_error = rng.multivariate_normal(mean=np.zeros(R.shape[0]), cov=R, size=(nfcst, int(da_window/da_tu)))
obs_truth = z_evolve[:,:,::obs_interval_real,::obs_interval_real]
obs = obs_truth + obs_error.reshape(obs_truth.shape)

## generate forecast truth
z_truth = np.zeros((nfcst,int(fcst_tu/ver_tu)+1,nx,ny),dtype=np.float32)
z_truth[:,0,:,:] = z_evolve[:,-1,:,:]
for ifcst in range(nfcst):
    zt = ft(z_evolve[ifcst,-1,:,:])
    t = 0.; i = 0
    for j in range(int(fcst_tu/dt)):
        t = (j + 1) * dt
        zt = model.bve_propagator(zt,forcet,verbose=True)
        if (j + 1) % int(round(ver_tu/dt)) == 0:
            z_truth[ifcst,i+1] = ift(zt)
            i += 1

## generate forecast initial state.
# gaussian noise.
phys_noise = rng.normal(0, 1, size=(nfcst, nx, ny))
spec_noise = (np.fft.fft2(phys_noise, axes=(-2, -1))/ np.sqrt(nx * ny)) 
# perturbation in spectral domain
spectral_perturbation = initial_coeff * spec_noise * std_spec
#B_init = initial_coeff*C
#initial_error = rng.multivariate_normal(mean=np.zeros(B_init.shape[0]), cov=B_init, size=nfcst)
z_noda = z_truth[:,0,:,:] + ift(spectral_perturbation)

## store
data_dict = {
    'zphy': {
        'data': zphy,
        'description': 'long-time simulation in physical space [n,nx,ny]'
    },
    'zbase': {
        'data': zbase,
        'description': 'long-time simulation in spectral space [n,nx,ny]'
    },
    'C_realization': {
        'data': zphy[:K//2],
        'description': 'an ensemble that stores the information of C.'
    },
}

with open('climate_simulation.pkl', 'wb') as f:
    pickle.dump(data_dict, f)

data_dict = {
    'z_initial': {
        'data': z_initial,
        'description': 'State at the start of assimilation window'
    },
    'B_init': {
        'data': zphy_anom/sqrt(K-1),
        'description': 'B_init matrix B_init.T@B_int + lamc*I'
    },
    'lam_B_init': {
        'data': lamc,
        'description': 'lambda to regularize B_init matrix B_init.T@B_int + lamc*I'
    },
    'Breg': {
        'data': error_center/sqrt(K//2-1),
        'description': 'Background matrix generated by error_center.T@error_center + lam*I [nsample,nx*ny]'
    },
    'lamb': {
        'data': lam,
        'description': 'lambda to regularize background matrix'
    },
    'obs': {
        'data': obs,
        'description': 'Observation'
    },
    'R': {
        'data': R,
        'description': 'Representative error vector.'
    }
}

with open('da_processing.pkl', 'wb') as f:
    pickle.dump(data_dict, f)

data_dict = {
    'z_truth': {
        'data': z_truth,
        'description': 'real value of forecast [nfcst,date,nx,ny]'
    },
    'z_noda': {
        'data': z_noda,
        'description': 'initial state without DA. as background field.'
    }
}

with open('forecast_info.pkl', 'wb') as f:
    pickle.dump(data_dict, f)