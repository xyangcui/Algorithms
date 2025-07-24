import numpy as np
from fun_extremal_index_sueveges import fun_extremal_index_sueveges

def fun_dynsys_univariate_analysis(x, quanti):
    """
    Computation of D1 and theta for each observation in the trajectory x, for
    a given quantile "quanti"


    REFERENCES
    Please cite:

    Davide Faranda, Gabriele Messori, Pascal Yiou. 2020. Diagnosing concurrent 
    drivers of weather extremes: application to hot and cold days in North 
    America, Climate Dynamics, 54, 2187-2201. doi: 10.1007/s00382-019-05106-3

    Davide Faranda, Gabriele Messori and Pascal Yiou. 2017. Dynamical proxies 
    of North Atlantic predictability and extremes. Scientific Reports, 7, 
    41278, doi: 10.1038/srep41278


    INPUTS
    x: a series of observations, a matrix arranged as [TIMExSPACE]
    quanti: a quantile for the selection of the recurrences


    OUTPUTS%
    D1: the series of local dimension, vector of size [TIME]
    theta: the series of local inverse persistences, vector of size [TIME]
    """

    # Computation of D1 and theta
    print('Computing dynamical quantities')
    D1    = np.zeros(x.shape[0])
    theta = np.zeros(x.shape[0])

    for j in range(x.shape[0]):
        # Compute the observables
        distance = np.linalg.norm(x[j, :] - x, axis=1)

        logdista = -np.log(distance)
        # Extract the threshold corresponding to the quantile defined   
        thresh = np.quantile(logdista, quanti)

        # Compute the extremal index, use the external function extremal_Sueveges
        theta[j] = fun_extremal_index_sueveges(logdista, quanti, thresh)

        #Sort the time series and find all the PoTs
        findid  = np.where(logdista > thresh)

        logextr = logdista[findid]

        #Extract the GPD parameters; since the distribution is exponential, the 
        #average of the PoTs is the unbiased estimator, which is just the mean 
        #of the exceedances.
        #The local dimension is the reciprocal of the exceedances of the PoTs
        D1[j] = 1/np.nanmean(logextr-thresh)

    return D1, theta
