import numpy as np

def fun_extremal_index_sueveges(Y, p, u):
    """
    This function computes the extremal index for a time series Y, given quantile p and a threshold u
    
    Reference:
    - Süveges, Mária. 2007. Likelihood estimation of the extremal index. Extremes, 10.1-2, 41-55, doi: 10.1007/s10687-007-0034-2
    
    Arguments:
    Y: time series of observations
    p: quantile for the computation of extreme value theory
    u: threshold estimated via the quantile (optional)
    
    Returns:
    theta: extremal index computed with the Sueveges formula
    """
    
    # Compute theta
        
    q  = 1 - p
    Li = np.where(Y > u)[0]
    Ti = np.diff(Li)
    Si = Ti - 1
    Nc = len(np.where(Si > 0)[0])
    N  = len(Ti)
    
    theta = (np.sum(q * Si) + N + Nc - np.sqrt( (np.sum(q * Si) + N + Nc)**2 - 8*Nc*np.sum(q*Si)) )/(2*np.sum(q*Si))

    return theta
