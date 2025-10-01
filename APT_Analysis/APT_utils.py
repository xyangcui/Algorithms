import numpy as np
import xarray as xr
from scipy.linalg import eigh

def APT(model_data,eofs=None,dtau=None,weights=None,eval=None,lead_dim_name='lead',ens_dim_name='member',time_dim_name='time',space_dim_names=['lat','lon']):
    """APT(model_data,eofs=None,dtau=None,weights=None,lead_dim_name='lead',ens_dim_name='member',time_dim_name='time',space_dim_names=['lat','lon'])
    function that maximizes Average Predictability Time (APT)

    required input:
    model_data: xarray DataArray with dimensions (in any order) [time, lead, ensemble member, lat, lon]

    optional inputs:
    eofs: eof patterns stored in xarray DataArray with dimensions [mode,lat,lon]
    ---> if excluded, the covariance matrices are computed without data reduction (which is both time consuming
        and likely to encounter error if the spatial dimension is larger than the "state" dimension)
    dtau: the amount of time separating different lead times (could be days in each month, for example)
    ---> if excluded, 1.0 assumed for all leads
    weights: currently excluded, please ignore and preweight model_data manually (probably with sqrt(cos(lat))
    lead_dim_name: the name of the lead_time/initialization dimension of model_data
    ---> if excluded, 'lead' assumed
    ens_dim_name: the name of the ensemble member dimension of model_data
    ---> if excluded, 'member' assumed
    time_dim_name: the name of the time dimension of model_data
    ---> if excluded, 'time' assumed
    space_dim_names: list array containing the spatial dimension names (2 elements expected)
    ---> if excluded, ['lat','lon'] assumed

    outputs: 
    predictable patterns, predictable variates, APT (eigenvalues), projection patterns"""

    # dtau represents the duration separating each lead, if not provided, 1.0 assumed
    #------------------------------------------------------------------------

    # if there are no dtau inputs default to 1.0
    if np.any(dtau == None):
        dtau = np.ones(np.size(model_data[lead_dim_name]))

    # if basis functions are provided, reduce the dimensionality using them
    #---------------------------------------------------------------------

    # if there are no eof inputs, then no dimension reduction
    if np.any(eofs == None):

        # move model data to [space x ensemble-lead-time] dimensions
        y_ec_reduced = model_data.stack(space=(space_dim_names),ec=(lead_dim_name,ens_dim_name,time_dim_name)).values

    # otherwise, reduce dimensions using eof inputs
    else:
        # first move model data to [space x ensembe-lead-time] dimensions
        y_ec = model_data.stack(space=(space_dim_names),ec=(lead_dim_name,ens_dim_name,time_dim_name))

        # move eofs to same to dimensions [space x mode]
        E = eofs.stack(space=(space_dim_names)).transpose('space','evn')

        # preallocate xarray with dimensions we want: space reduced to Neofs
        y_ec_reduced = 0.0 * y_ec[:np.size(E.evn),:]

        # now reduce dimensionality by projecting EOFs onto the original data
        y_ec_reduced.values = np.linalg.inv(E.values.transpose() @ E.values) @ E.values.transpose() @ y_ec.values

    # get the signal covariance matrix and the noise covariance matrix
    #---------------------------------------------------------------------

    # unstack dimensions condition/time vs ensemble for matrix calculations
    y_reduced = y_ec_reduced.unstack('ec').transpose(lead_dim_name,ens_dim_name,time_dim_name,'space')

    # get the total covariance matrix (overall variance)
    sigma_inf = np.cov(y_ec_reduced)

    # preallocate noise covariance matrix (one for each initialization)
    sigma_tau = np.zeros([np.size(model_data[lead_dim_name]),np.shape(sigma_inf)[0],np.shape(sigma_inf)[1]])

    # iterate to compute the noise covariance (time mean ensemble variance) at each lead time (weighted by dtau)
    for j in range(0,np.size(model_data[lead_dim_name])):
        sigma_tau[j,:,:] = np.mean([np.cov(y_reduced[j,:,i,:].transpose()) for i in range(np.size(model_data[time_dim_name]))], axis=0) * dtau[j]

    # this is the parameter we are optimizing
    G = 2.0 * (sigma_inf * np.sum(dtau) - np.sum(sigma_tau,axis=0))

    # Solve the generalized eigenvalue problem
    #------------------------------------------------------------------------    
    # This will get the eigenvalues, l, and eigenvectors, Q, with dimensions [space x mode] for Q
    # ---> That's because the operation is, signal_cov @ Q = L @ Q^T @ noise_cov @ Q, which implies
    # ---> [space x space] @ [space x mode] = [mode x mode] @ [mode x space] @ [space x space] @ [space x mode]  

    # get the eigenvectors and eigenvalues using scipy function eigh
    lasc, Qasc = eigh(G, sigma_inf)

    # By default, eigh returns eigenvectors in ascending order; we reverse this
    l = lasc[::-1]
    Q = Qasc[:,::-1]

    # normalization factor is based on total covariance matrix (following DelSole table 18.2)
    Gamma = np.diag(Q.transpose() @ sigma_inf @ Q)
    Norm = np.diag(1.0 / np.sqrt(Gamma))

    # normalize Q
    Qnorm = Q @ Norm

    # predictable patterns | following DelSole and Tippett textbook
    P_Espace = sigma_inf @ Qnorm
    if eofs = None:
        P = P_Espace  
    else:
        P = E.values @ P_Espace
    # predictable variates | one for each ensemble member and initialization
    PV = y_reduced.values @ Qnorm

    # postprocessing: for output as xarrays
    #------------------------------------------------------------------------

    # coordinate arrays for the output data
    mode = np.arange(1,np.size(y_ec_reduced.space) + 1,1)
    lat = model_data[space_dim_names[0]]
    lon = model_data[space_dim_names[1]]
    time = model_data[time_dim_name]
    member = model_data[ens_dim_name]
    lead = model_data[lead_dim_name]

    # Create an empty DataArray with the specified dimensions and coordinates
    patterns = xr.DataArray(
        np.zeros(( np.size(mode), np.size(lat), np.size(lon) )),  # Fill with zeros initially
        dims=["mode", space_dim_names[0], space_dim_names[1]],
        coords={"mode": mode, space_dim_names[0]: lat, space_dim_names[1]: lon},
    )

    # these are the patterns that when projected onto the data, produce the predictable variates
    q_patterns = patterns * 1.0

    # arrange the patterns in an xarray
    patterns_stacked = 0.0 * patterns.rename('predictable_patterns').stack(space=(space_dim_names))
    patterns_stacked.values = P.transpose() # without transpose, dimensions are space x mode
    patterns = patterns_stacked.unstack()

    #un weighted.
    if eofs == None:
        patterns = patterns
    else:
        patterns = patterns/weights

    # add attrs of variance y_ec_reduced[neofs,ntimes]; P_Espace[neofs,mode]
    variance = np.mean([np.dot(y_ec[:,i].values.transpose(),y_ec[:,i].values) for i in range(y_ec.values.shape[1])],axis=0)
    varratio = np.dot(P.transpose(),P).diagonal() / variance
    patterns.attrs['ratio'] = varratio
    print(varratio)

    # arrange the patterns in an xarray
    q_patterns_stacked = 0.0 * q_patterns.rename('projection_patterns').stack(space=(space_dim_names))
    if eofs == None:
        q_patterns_stacked.values = Qnorm.transpose() # without transpose, dimensions are space x mode
    else:
        q_patterns_stacked.values = (E.values @ Qnorm).transpose() # without transpose, dimensions are space x mode

    q_patterns = q_patterns_stacked.unstack()

    # Create an empty DataArray with specified dimensions and coordinates
    predictable_variates = xr.DataArray(
        np.zeros(( np.size(lead), np.size(member), np.size(time), np.size(mode) )),
        dims=[lead_dim_name, ens_dim_name, time_dim_name, "mode"],
        coords={lead_dim_name: lead, ens_dim_name: member, time_dim_name: time,  "mode": mode},
    )

    # signal to noise ratio from eigenvalues
    apt = xr.DataArray(data=l,dims=["mode"],coords={"mode": mode})
    apt = apt.rename('apt')
    # fill the predictable variates array
    predictable_variates.values = PV
    predictable_variates = predictable_variates.rename('v')
    # return the patterns and predictable variates
    return patterns, predictable_variates, apt, q_patterns, P_Espace