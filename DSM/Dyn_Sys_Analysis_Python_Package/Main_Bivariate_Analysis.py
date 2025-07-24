import os

import numpy as np
import xarray as xr

from fun_dynsys_bivariate_analysis import fun_dynsys_bivariate_analysis

"""
INFO
This code computes the local dimension, persistent and the co-recurrences
coefficient of two tensors. 

REFERENCES
Please cite:

Davide Faranda, Gabriele Messori, Pascal Yiou. 2020. Diagnosing concurrent 
drivers of weather extremes: application to hot and cold days in North 
America, Climate Dynamics, 54, 2187-2201. doi: 10.1007/s00382-019-05106-3

Davide Faranda, Gabriele Messori and Pascal Yiou. 2017. Dynamical proxies 
of North Atlantic predictability and extremes. Scientific Reports, 7, 
41278, doi: 10.1038/srep41278 


INPUTS
x_lonlat and y_lonlat: tensors. First index is time, second and third 
indices are lon-lat, which should also be provided as input.
quanti: Percentile determining the radius of the sphere in the space of
distances.


OUTPUTS
From this code we extract the following time series:
Inverse persistence theta_x, theta_y and co-persistence theta_con
Local dimensions D1_x, D1_y and c-odimension D1_con
Co-recurrence coefficient alpha
They have all dimension [time]
"""
# INPUTS
# x_lonlat and y_lonlat are tensors containing dimensions [time,lon,lat]. 
# The following examples uses random tensors, drawn from a normal distribution 
# Please change with the loading part of your fields from, e.g. GRIB or NC files

quanti = 0.98

fdir   = os.getenv('dir')
xname  = os.getenv('xf')
yname  = os.getenv('yf')
ouname = os.getenv('ouname')
xvar   = os.getenv('xvar')
yvar   = os.getenv('yvar')

xf = xr.open_dataset(fdir+'/'+xname)
yf = xr.open_dataset(fdir+'/'+yname)

# Generate random 'x' and 'y' tensors
x_lonlat = xf[xvar]
y_lonlat = yf[yvar]

x_lonlat = np.array(x_lonlat)
y_lonlat = np.array(y_lonlat)
# Define time, lon, and lat
time = xf['time']
lon  = xf['lon']
lat  = xf['lat']

# Reshape tensors to obtain time * space matrices
x = x_lonlat.reshape(len(time), len(lon) * len(lat))
y = y_lonlat.reshape(len(time), len(lon) * len(lat))

# Compute dynamical quantities
D1_x, D1_y, D1_con, theta_x, theta_y, theta_con, alpha = fun_dynsys_bivariate_analysis(x, y, quanti)

# Output to a netcdf file.
ds = xr.Dataset({'D1_x': (('time'), D1_x)}, {'D1_y': (('time'), D1_y)}, {'D1_con': (('time'), D1_con)},\
    {'theta_x': (('time'), theta_x)}, {'theta_y': (('time'), theta_y)}, {'theta_con': (('time'), theta_con)},\
        {'alpha': (('time'), alpha)}, coords={'time': time})

ds.to_netcdf(fdir+'/'+ouname)
