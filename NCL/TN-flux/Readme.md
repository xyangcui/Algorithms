subroutine tnf_xy(psiam,um,vm,dlati,dloni,mlat,mlon,prem,neqlats,neqlatn,tnf)

Wave-activity flux based on Eq (38) of Takaya and Nakamura (2001) for horizontal compoents

 Takaya, K., Nakamura, H., 2001: 
  A formulation of a phase-independent wave-activity flux for stationary and migratory quasi-geostrophic eddies on zonally-varying basic flow. J. Atmos. Sci., 58, 608--627. 
 
 !!! Input file format : because fortran change from first to last, netcdf from last to first. so here in netcdf format need (lat,lon) !!!
 
  1-level 
  
 psiam : psi anom  
 
 um: back u 
 
 vm: back v 
 
 dlati,dloni: interval of lat and lon.
 
 mlat,mlon: x
 
 prem: prem=p/1000 hPa
 
 neqlats: 10S's indice
 
 neqlatn: 10N's indice
 
 tnf: (2,mlon,mlat) tnf(1) is fx, tnf(2) is fy.

 Output data
 
   x,y component 
   
   (361x181) North to South. 
   
!! Note that it denots (lonxlat), please convert!

   p = p/1000.
   
The limit latitudes of QG approximation

( This may be changed)

