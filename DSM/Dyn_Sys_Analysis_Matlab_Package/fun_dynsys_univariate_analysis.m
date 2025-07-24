function [D1, theta] = fun_dynsys_univariate_analysis(x, quanti)

%Computation of D1 and theta for each observation in the trajectory x, for
%a given quantile "quanti"


%REFERENCES%
%Please cite:

%Davide Faranda, Gabriele Messori, Pascal Yiou. 2020. Diagnosing concurrent 
%drivers of weather extremes: application to hot and cold days in North 
%America, Climate Dynamics, 54, 2187-2201. doi: 10.1007/s00382-019-05106-3

%Davide Faranda, Gabriele Messori and Pascal Yiou. 2017. Dynamical proxies 
%of North Atlantic predictability and extremes. Scientific Reports, 7, 
%41278, doi: 10.1038/srep41278


%INPUTS%
%x: a series of observations, a matrix arranged as [TIMExSPACE]
%quanti: a quantile for the selection of the recurrences


%OUTPUTS%
%D1: the series of local dimension, vector of size [TIME]
%theta: the series of local inverse persistences, vector of size [TIME]


%% Computation of D1 and theta
disp('Computing dynamical quantities')
D1 = zeros(size(x,1),1);

for j=1:size(x,1)

    % Compute the observables
    logdista=-log(pdist2(x(j,:),x));
    % Extract the threshold corresponding to the quantile defined   
    thresh=quantile(logdista, quanti);

    % Compute the extremal index, use the external function extremal_Sueveges
    theta(j)=fun_extremal_index_sueveges(logdista,quanti, thresh);

    %Sort the time series and find all the PoTs
    logextr=logdista(logdista>thresh);
    logextr=logextr(isfinite(logextr));


    %Extract the GPD parameters; since the distribution is exponential, the 
    %average of the PoTs is the unbiased estimator, which is just the mean 
    %of the exceedances.
    %The local dimension is the reciprocal of the exceedances of the PoTs
    D1(j,1)=1./mean(logextr-thresh);
end
