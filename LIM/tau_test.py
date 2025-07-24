import pickle, xarray, numpy
from scipy.linalg import expm
from math import sqrt
import matplotlib.pyplot as plt
from   matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.ticker as ticker

n = 8  #EOFs of U200
lats = 10
latn = 80
lonw = 120
lone = 240

ddir = "/home/sunming/data5/cuixy/Subpre_NPJ/data"

with open(ddir+'/LIM_params.pkl', 'rb') as file:
    data = pickle.load(file)

c0 = data['c0']   #lag-0 covariance matrix
L  = data['L']    #exp(L*tau)

#convert pcs skill into real skill. skill*EOFs
#eofs = xarray.open_dataarray(ddir+'/eof_u200.nc')
#eofs = eofs.loc[:n,lats:latn,lonw:lone]
#eval = eofs.attrs.get('eval'); eval = numpy.sqrt(eval[:n])
#eofs = eofs*eval[:, numpy.newaxis, numpy.newaxis]
#eof  = eofs.to_numpy().reshape(eofs.shape[0],-1)

#u = xarray.open_dataarray(ddir+"/LIM_obs_u200_v2.nc")
#u = u.loc[:,:,lats:latn,lonw:lone]
#lat = u.lat; lon = u.lon
#X = u.sel(time=u.time.dt.month.isin([12, 1, 2, 3]))
#X = u[:,0,:,:].to_numpy().reshape(u.shape[0],-1) 

x = xarray.open_dataarray(ddir+"/state_vectors.nc")
x = x.to_numpy()

Tau = numpy.arange(1,31,1)
tr  = numpy.zeros([2,Tau.shape[0]])

for i,tau in enumerate(Tau):
    #original
    x0 = numpy.copy(x[:,:-tau])
    xt = numpy.copy(x[:,tau:])
    ct = numpy.dot(xt, x0.T)/(x0.shape[1]-1)
    tr[0,i] = numpy.trace(ct)
    #LIM prediction
    F = numpy.dot(expm(L.real*tau),c0)
    tr[1,i] = numpy.trace(F)

# Set up plot 
fig, ax = plt.subplots()
fig.set_size_inches(12,8)

# Plot each line for various choices of tau0 
colors = ["red","blue"]
labels = ["OBS-based","LIM-based"]
for iT0 in range(2):
    ax.plot(Tau, tr[iT0,:],label=labels[iT0],color=colors[iT0])
ax.set_xlabel(r'$\tau_{0}$',fontsize=16)
ax.set_ylabel('Norm',fontsize=16)
ax.set_title('tau_test',fontsize=16)
ax.set_xlim([1,30])
ax.set_xticks(ticks=[1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30])
#ax1.set_ylim([0,yupper])
ax.tick_params(labelsize=14)
#ax1.grid()
plt.legend()
plt.show()
