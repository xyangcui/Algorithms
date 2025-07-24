function [D1_x, D1_y, D1_con, theta_x, theta_y, theta_con, alpha] = ...
    fun_dynsys_bivariate_analysis(x, y, quanti)

%Computation of D1, theta and alpha for each observation in the bivariate 
%trajectory x,y for a given quantile "quanti"


%REFERENCES%
%Please cite:

%Davide Faranda, Gabriele Messori, Pascal Yiou. 2020. Diagnosing concurrent 
%drivers of weather extremes: application to hot and cold days in North 
%America, Climate Dynamics, 54, 2187-2201. doi: 10.1007/s00382-019-05106-3

%Davide Faranda, Gabriele Messori and Pascal Yiou. 2017. Dynamical proxies 
%of North Atlantic predictability and extremes. Scientific Reports, 7, 
%41278, doi: 10.1038/srep41278


%INPUTS%
%x,y: series of observations, matrices arranged as [SPACExTIME]. Note that
%for the bivariate case, the two matrices must have the same number of
%timesteps.
%quanti: a quantile for the selection of the recurrences


%OUTPUTS%
%D1_x,D1_y,D1_con: the series of local dimension for each variable 
%individually and the codimension for the two variables jointly, vectors of 
%size [TIME]
%theta_x,theta_y, theta_con: the series of local inverse persistences for 
%each variable individually and the copersistence for the two variables 
%jointly, vectors of size [TIME]
%alpha: the corecurrence coeffient


%% Computation of D1_x, D1_y, D1_con, theta_x, theta_y, theta_con, alpha
disp('Computing dynamical quantities')
D1_x = zeros(size(x,1),1);
D1_y = zeros(size(x,1),1);
D1_con = zeros(size(x,1),1);
theta_x = zeros(size(x,1),1);
theta_y = zeros(size(x,1),1);
theta_con = zeros(size(x,1),1);
alpha = zeros(size(x,1),1);

for j=1:size(x,1)
   
    %For each map at time (j) compute the distances for the two variables
    %separately
    distance_x = pdist2(x(j,:),x);
    distance_y = pdist2(y(j,:),y);

    
    % Renormalize the distances  by their norm
    distance_x =distance_x/norm(distance_x);
    distance_y =distance_y/norm(distance_y);
    distance_con= sqrt(distance_x.^2+(distance_y.^2));

    %Compute the observables g=-log(dist(---)) 
    logdista_x=-log(distance_x);
    logdista_y=-log(distance_y);
    logdista_con=-log(distance_con);
    
    %Compute the threshold for x,y, and the joint threshold. This defines
    %the radius of the ball in phase space to obtain the recurrences of the
    %map under examination
    thresh_x=quantile(logdista_x, quanti);
    thresh_y=quantile(logdista_y, quanti);
    thresh_con=quantile(logdista_con, quanti);
    
    %Compute the extremal index using the function extremal_Sueveges
    theta_x(j,1)=fun_extremal_index_sueveges(logdista_x,quanti);
    theta_y(j,1)=fun_extremal_index_sueveges(logdista_y,quanti);
    theta_con(j,1)=fun_extremal_index_sueveges(logdista_con,quanti);

    %Select the recurrences for x,y and the corecurrences x-y in the neighborhood defined by the quantile quanti
    findidx_x=find(logdista_x>thresh_x & ~isinf(logdista_x));
    findidx_y=find(logdista_y>thresh_y & ~isinf(logdista_y));
    findidx_con=find(logdista_con>thresh_con & ~isinf(logdista_con));
    findidx=find(logdista_x>thresh_x & logdista_y>thresh_y & ~isinf(logdista_x) & ~isinf(logdista_y));

    %Select the distances that match the previous conditions
    logextr_x = logdista_x(findidx_x);
    logextr_y = logdista_y(findidx_y);
    logextr_con = logdista_con(findidx_con);
  
    %Extract the local dimensions;
    D1_x(j,1) = 1./nanmean(logextr_x-thresh_x);
    D1_y(j,1) = 1./nanmean(logextr_y-thresh_y);
    D1_con(j,1) = 1./nanmean(logextr_con-thresh_con);
    %Compute the corecurrence_coefficient
    alpha(j,1)=length(findidx)./length(findidx_x);
end  
