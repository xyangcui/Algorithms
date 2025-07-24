This package contains code to estimate the local dimension, persistence and co-recurrence of dynamical systems from time series. It contains two main scripts and three
auxiliary functions.


Main scripts:

-Main_Univariate_Analysis_Example.m : It includes the computation of the local dimension and persistence for the Lorenz 1963 attractor in the univariate case. 

-Main_Bivariate_Analysis_Example.m : It includes the computation of the local dimension, the persistence and the co-recurrence coefficient for two random fields in the
bivariate case.


Auxiliary functions:

-fun_dynsys_univariate_analysis.m : Computation of local dimensions and persistence for each point along a trajectory x, for a given quantile.

-fun_dynsys_bivariate_analysis.m : Computation of local dimensions, persistence, co-dimensions, co-persistence, co-recurrences for each point along a trajectory x, for 
a given quantile.

-fun_extremal_index_sueveges : This function computes the extremal index for a time series. 


The code is distributed under the CC BY-NC 3.0 license. You can find the full terms and conditions here: https://creativecommons.org/licenses/by-nc/3.0/
Please contact davide.faranda@cea.fr for more information. 


Whenever you use this code, please cite:

Davide Faranda, Gabriele Messori, Pascal Yiou. 2020. Diagnosing concurrent drivers of weather extremes: application to hot and cold days in North America, 
Climate Dynamics, 54, 2187-2201. doi: 10.1007/s00382-019-05106-3

Davide Faranda, Gabriele Messori and Pascal Yiou. 2017. Dynamical proxies of North Atlantic predictability and extremes. Scientific Reports, 7, 41278, 
doi: 10.1038/srep41278


For the estimator of the persistence, please cite:

Süveges, Mária. 2007. Likelihood estimation of the extremal index. Extremes, 10.1-2, 41-55, doi: 10.1007/s10687-007-0034-2
