# Dynamical system metrics

The original code is matlab code, which is coded by Faranda.<br> 
A python code is still given.<br>
<br>
This package contains code to estimate the local dimension, persistence and co-recurrence of dynamical systems from time series. It contains two main scripts and three
auxiliary functions.<br>

Main scripts:<br>

-Main_Univariate_Analysis_Example.m : It includes the computation of the local dimension and persistence for the Lorenz 1963 attractor in the univariate case.<br> 

-Main_Bivariate_Analysis_Example.m : It includes the computation of the local dimension, the persistence and the co-recurrence coefficient for two random fields in the<br>
bivariate case.<br>


Auxiliary functions:<br>

-fun_dynsys_univariate_analysis.m : Computation of local dimensions and persistence for each point along a trajectory x, for a given quantile.<br>

-fun_dynsys_bivariate_analysis.m : Computation of local dimensions, persistence, co-dimensions, co-persistence, co-recurrences for each point along a trajectory x, a given quantile.<br> 

-fun_extremal_index_sueveges : This function computes the extremal index for a time series.<br> 

The code is distributed under the CC BY-NC 3.0 license. You can find the full terms and conditions here: https://creativecommons.org/licenses/by-nc/3.0/<br>
Please contact davide.faranda@cea.fr for more information.<br> 

Whenever you use this code, please cite:<br>
<br>
Davide Faranda, Gabriele Messori, Pascal Yiou. 2020. Diagnosing concurrent drivers of weather extremes: application to hot and cold days in North America, <br>
Climate Dynamics, 54, 2187-2201. doi: 10.1007/s00382-019-05106-3<br>
<br>
Davide Faranda, Gabriele Messori and Pascal Yiou. 2017. Dynamical proxies of North Atlantic predictability and extremes. Scientific Reports, 7, 41278, <br>
doi: 10.1038/srep41278<br>
<br>
<br>
For the estimator of the persistence, please cite:<br>
<br>
Süveges, Mária. 2007. Likelihood estimation of the extremal index. Extremes, 10.1-2, 41-55, doi: 10.1007/s10687-007-0034-2
<br>
![figure](382_2019_5106_Fig1_HTML.webp#pic_center)
