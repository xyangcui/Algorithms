
%INFO%
%Example of the computation of local dimensions and persistence for the
%Lorenz 1963 equations


%REFERENCES%
%Please cite:

%Davide Faranda, Gabriele Messori, Pascal Yiou. 2020. Diagnosing concurrent 
%drivers of weather extremes: application to hot and cold days in North 
%America, Climate Dynamics, 54, 2187-2201. doi: 10.1007/s00382-019-05106-3

%Davide Faranda, Gabriele Messori and Pascal Yiou. 2017. Dynamical proxies 
%of North Atlantic predictability and extremes. Scientific Reports, 7, 
%41278, doi: 10.1038/srep41278 


%INPUT%
%quanti: Percentile determining the radius of the sphere in the space of
%distances.


%OUTPUTS%
%From this code we extract the following time series:
%Inverse persistence theta
%Local dimensions D1
%They both have dimension [time]


clear;
close all;

%% INPUTS
quanti = 0.98;

%% LORENZ ATTRACTOR SIMULATION
%Define the integration total time T and the time-step dt;
T = 1000000;
dt= 0.01;
t= 0:dt:T;
%Lorenz attractor parameter
beta=8/3; sigma=10; rho=28;
%Lorenz Initial Conditions
x(1) = 1; y(1) = 1; z(1) = 1;
%Iterate Lorenz attractor with an Euler Scheme
for i =1:(T./dt) 
x(i+1)= x(i) + dt.*(sigma*(y(i)-x(i))); 
y(i+1)= y(i) + dt.*(x(i)*(rho - z(i))-y(i)); 
z(i+1)= z(i) + dt.*(x(i)*y(i) - beta*z(i)); 
end

%% DIMENSION AND PERSISTENCE COMPUTATION
%rearrange the trajectory in a single matrix [TIMExSPACE]
data=[x' y' z'];
[D1, theta]=fun_dynsys_univariate_analysis(data,quanti);

%% EXAMPLE PLOTS
%Plot a section of the Lorenz attractor with D1 and theta in colors
figure
subplot(2,1,1)
scatter(x,z,10,D1, 'filled')
caxis([0 3]), colorbar; box on;
title(['Local dimensions, Average D=', num2str(nanmean(D1))])
xlabel('x')
ylabel('z')

subplot(2,1,2)
scatter(x,z,10,theta, 'filled')
caxis([0 0.3]); colorbar; box on;
title(['Local inverse persistence, Average \theta=', num2str(nanmean(theta))])
xlabel('x')
ylabel('z')
