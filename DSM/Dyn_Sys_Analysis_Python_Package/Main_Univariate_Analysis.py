import os

import numpy as np
import xarray as xr

from fun_dynsys_univariate_analysis import fun_dynsys_univariate_analysis

"""
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
"""

# INPUTS
# x_lonlat and y_lonlat are tensors containing dimensions [time,lon,lat]. 
# The following examples uses random tensors, drawn from a normal distribution 
# Please change with the loading part of your fields from, e.g. GRIB or NC files

quanti = 0.98

fdir   = os.getenv('dir')
xname  = os.getenv('xf')
ouname = os.getenv('ouname')
xvar   = os.getenv('xvar')

xf = xr.open_dataset(fdir+'/'+xname)

# Generate random 'x' and 'y' tensors
x_lonlat = xf[xvar]

x_lonlat = np.array(x_lonlat)

# Define time, lon, and lat
time = xf['time']
lon  = xf['lat']
lat  = xf['lon']

# Reshape tensors to obtain time * space matrices
x = x_lonlat.reshape(len(time), len(lon) * len(lat))

# Compute dynamical quantities
D1_x, theta_x = fun_dynsys_univariate_analysis(x,quanti)

# Output to a netcdf file.
ds = xr.Dataset({'D1': (('time'), D1_x)}, {'theta': (('time'), theta_x)}, coords={'time': time})

ds.to_netcdf(fdir+'/'+ouname)
