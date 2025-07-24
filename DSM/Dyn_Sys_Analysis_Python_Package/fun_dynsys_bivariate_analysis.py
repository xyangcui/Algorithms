import numpy as np
from fun_extremal_index_sueveges import fun_extremal_index_sueveges

def fun_dynsys_bivariate_analysis(x, y, quanti):
    """
    Computation of D1, theta and alpha for each observation in the bivariate trajectory x, y for a given quantile "quanti"
    
    References:
    - Davide Faranda, Gabriele Messori, Pascal Yiou. 2020. Diagnosing concurrent drivers of weather extremes: application to hot and cold days in North America, Climate Dynamics, 54, 2187-2201. doi: 10.1007/s00382-019-05106-3
    - Davide Faranda, Gabriele Messori and Pascal Yiou. 2017. Dynamical proxies of North Atlantic predictability and extremes. Scientific Reports, 7, 41278, doi: 10.1038/srep41278
    
    Arguments:
    x, y: series of observations, matrices arranged as [SPACE x TIME]. Note that for the bivariate case, the two matrices must have the same number of timesteps.
    quanti: a quantile for the selection of the recurrences
    
    Returns:
    D1_x, D1_y, D1_con: the series of local dimension for each variable individually and the codimension for the two variables jointly, vectors of size [TIME]
    theta_x, theta_y, theta_con: the series of local inverse persistences for each variable individually and the copersistence for the two variables jointly, vectors of size [TIME]
    alpha: the corecurrence coefficient
    """
    
    # Computation of D1_x, D1_y, D1_con, theta_x, theta_y, theta_con, alpha
    print('Computing dynamical quantities')
    D1_x = np.zeros(x.shape[0])
    D1_y = np.zeros(x.shape[0])
    D1_con = np.zeros(x.shape[0])
    theta_x = np.zeros(x.shape[0])
    theta_y = np.zeros(x.shape[0])
    theta_con = np.zeros(x.shape[0])
    alpha = np.zeros(x.shape[0])
    
    for j in range(x.shape[0]):
        # For each map at time (j), compute the distances for the two variables separately
        distance_x = np.linalg.norm(x[j, :] - x, axis=1)
        distance_y = np.linalg.norm(y[j, :] - y, axis=1)
        
        # Renormalize the distances by their norm
        distance_x = distance_x / np.linalg.norm(distance_x)
        distance_y = distance_y / np.linalg.norm(distance_y)
        distance_con = np.sqrt(distance_x**2 + distance_y**2)
        
        # Compute the observables g = -log(dist(---)) 
        logdista_x = -np.log(distance_x)
        logdista_y = -np.log(distance_y)
        logdista_con = -np.log(distance_con)
        
        # Compute the threshold for x, y, and the joint threshold. This defines the radius of the ball in phase space to obtain the recurrences of the map under examination
        thresh_x = np.quantile(logdista_x, quanti)
        thresh_y = np.quantile(logdista_y, quanti)
        thresh_con = np.quantile(logdista_con, quanti)
        
        # Compute the extremal index using the function extremal_Sueveges
        theta_x[j] = fun_extremal_index_sueveges(logdista_x, quanti, thresh_x)
        theta_y[j] = fun_extremal_index_sueveges(logdista_y, quanti, thresh_y)
        theta_con[j] = fun_extremal_index_sueveges(logdista_con, quanti, thresh_con)
        
        # Select the recurrences for x, y and the corecurrences x-y in the neighborhood defined by the quantile quanti
        findidx_x = np.where((logdista_x > thresh_x) & (~np.isinf(logdista_x)))[0]
        findidx_y = np.where((logdista_y > thresh_y) & (~np.isinf(logdista_y)))[0]
        findidx_con = np.where((logdista_con > thresh_con) & (~np.isinf(logdista_con)))[0]
        findidx = np.where((logdista_x > thresh_x) & (logdista_y > thresh_y) & (~np.isinf(logdista_x)) & (~np.isinf(logdista_y)))[0]
        
        # Select the distances that match the previous conditions
        logextr_x = logdista_x[findidx_x]
        logextr_y = logdista_y[findidx_y]
        logextr_con = logdista_con[findidx_con]
        
        # Extract the local dimensions
        D1_x[j] = 1 / np.nanmean(logextr_x - thresh_x)
        D1_y[j] = 1 / np.nanmean(logextr_y - thresh_y)
        D1_con[j] = 1 / np.nanmean(logextr_con - thresh_con)
        
        # Compute the corecurrence coefficient
        alpha[j] = len(findidx) / len(findidx_x)
    
    return D1_x, D1_y, D1_con, theta_x, theta_y, theta_con, alpha
