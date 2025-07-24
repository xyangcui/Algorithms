
%INFO%
%This code computes the local dimension, persistent and the co-recurrences
%coefficient of two tensors. 

%REFERENCES%
%Please cite:

%Davide Faranda, Gabriele Messori, Pascal Yiou. 2020. Diagnosing concurrent 
%drivers of weather extremes: application to hot and cold days in North 
%America, Climate Dynamics, 54, 2187-2201. doi: 10.1007/s00382-019-05106-3

%Davide Faranda, Gabriele Messori and Pascal Yiou. 2017. Dynamical proxies 
%of North Atlantic predictability and extremes. Scientific Reports, 7, 
%41278, doi: 10.1038/srep41278 


%INPUTS%
%x_lonlat and y_lonlat: tensors. First index is time, second and third 
%indices are lon-lat, which should also be provided as input.
%quanti: Percentile determining the radius of the sphere in the space of
%distances.


%OUTPUTS%
%From this code we extract the following time series:
%Inverse persistence theta_x, theta_y and co-persistence theta_con
%Local dimensions D1_x, D1_y and c-odimension D1_con
%Co-recurrence coefficient alpha
%They have all dimension [time]

clear
close all;


%% INPUTS 

quanti=0.98;

%x_lonlat and y_lonlat are tensors containing dimensions [time,lon,lat]. 
%The following examples uses random tensors, drawn from a normal distribution 
%Please change with the loading part of your fields from, e.g. GRIB or NC files

%load or define a random 'x' field
x_lonlat=randn(1000,10,15);
%load or define a random 'y' field
y_lonlat=randn(1000,10,15);
%load or define time
time=1:1000;
%define lon;
lon=1:10;
%define lat;
lat=1:15;


% Reshape to obtain time * space matrices
x=reshape(x_lonlat, length(time), length(lon)*length(lat));
y=reshape(y_lonlat, length(time), length(lon)*length(lat));


%Remove the original variables to free some RAM 
clear x_lonlat;
clear y_lonlat;

%Compute all the dynamical quantities
[D1_x,D1_y,D1_con,theta_x,theta_y, theta_con, alpha]=fun_dynsys_bivariate_analysis(x,y,quanti);
  
%% EXAMPLE PLOTS
%Plot the different monovariate and bivariate dynamical quantities
figure
subplot(3,1,1)
hold on
plot(time,D1_x)
plot(time,D1_y)
plot(time,D1_con)
title(['a) Local dimensions'])
xlabel('time')
ylabel('D1')
legend('x','y','co-dimension')

subplot(3,1,2)
hold on
plot(time,theta_x)
plot(time,theta_y)
plot(time,theta_con)
title(['b) Local inverse persistence'])
xlabel('time')
ylabel('\theta')
legend('x','y','co-persistence')

subplot(3,1,3)
hold on
plot(time,alpha)
title(['c) Co-recurrence coefficient'])
xlabel('time')
ylabel('\alpha')
