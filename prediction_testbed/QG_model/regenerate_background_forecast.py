import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from baro_utilis import BARO_VORT, ift, ft
import numpy as np
import pickle, json

## set model parameters
with open("config.json", "r") as f:
    config = json.load(f)
## model configration
dt = config['model']['dt']
nx = config['model']['nx']
ny = config['model']['ny']
nobs = config['observations']['nobs']
# cases of forecast
nfcst = config['forecast_evaluation']['n_cases']
# number of perturbed members
eda_members = config['eda']['default_members']

model = BARO_VORT(config['model'])

da_steps = config['data_assimilation']['four_d_var_window_steps']

with open('da_processing.pkl', 'rb') as f:
    data_dict = pickle.load(f)
# initial state of control forecast at the start of assimilation window
z_initial_control = data_dict['z_initial_control']['data']
# initial state of perturbed forecast at the start of assimilation window
z_initial_pert = data_dict['z_initial_pert']['data']

z_initial_control_spec = ft(z_initial_control)
z_initial_perturbed_spec = ft(z_initial_pert)

with open('climate_simulation.pkl', 'rb') as f:
    data_dict = pickle.load(f)
    forcet = data_dict['forcing']['data']

with open('forecast_info.pkl', 'rb') as f:
    data_dict = pickle.load(f)
z_truth = data_dict['z_truth']

z_noda = np.zeros((nfcst,1+eda_members,nx,ny),dtype=np.float32)
for ifcst in range(nfcst):
    z_control = z_initial_control_spec[ifcst]
    for i in range(int(da_steps)):
        z_noda[ifcst,0,:,:] = ift(model.bve_propagator(z_control,forcet,verbose=True))
    for j in range(eda_members):
        for i in range(int(da_steps)):
            z_pert = z_initial_perturbed_spec[ifcst,j]
            z_noda[ifcst,j+1,:,:] = ift(model.bve_propagator(z_pert,forcet,verbose=True))

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

with open('forecast_info_v2.pkl', 'wb') as f:
    pickle.dump(data_dict, f)