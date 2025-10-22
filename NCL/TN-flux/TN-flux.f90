subroutine tnf_xy(psiam,um,vm,dlati,dloni,mlat,mlon,prem,neqlats,neqlatn,tnf)

! Wave-activity flux based on Eq (38) of Takaya and Nakamura (2001)
!  for horizontal compoents

! Takaya, K., Nakamura, H., 2001: 
!   A formulation of a phase-independent wave-activity flux 
!   for stationary and migratory quasi-geostrophic eddies 
!   on zonally-varying basic flow. J. Atmos. Sci., 58, 608--627. 
 
! Input file format : (lat,lon)!! 
!  lon; 144 (0 - 357.5E)
!  lat;  73 (90S - 90N)
!  1-level 
! psiam : psi anom  um: back u vm: back v 

! Output data
!   x,y component 
!   (361x181) North to South. 
!! Note that it denots (lonxlat), please convert!
!   p = p/1000.
!   f=2*7.29E-5 *sin(psi)
! The limit latitudes of QG approximation
!  ( This may be changed)

  implicit none
! wait for put.
  integer :: mlon,mlat
  real :: dlati,dloni,prem  
  integer :: neqlats,neqlatn
  real :: um(mlon,mlat),vm(mlon,mlat),psiam(mlon,mlat),tnf(2,mlon,mlat)

! in values.  
  real,parameter :: unavl=9.999e+20,ubmin=1.,tday=60.*60.*24.,erd=6.371E+06 
  real :: u(mlon,mlat),v(mlon,mlat)
  integer ilon,ilone,ilonw,ilat,ilatn,ilats
  real :: pai,dgrd,rloni,rloni2, rlati,rlati2,oum2
  real :: termxU,termxV,termyU,termyV
  real :: wspeed
  real :: dudx,dvdx,dudy,dvdy
  real :: rlat,coriol,coslat
  real :: uamn,uams,uame,uamw,vame,vamw,psiamn,psiams,psiame,psiamw
  real :: fx,fy,psia,uam,vam,ub,vb

! constant
  pai   = acos(-1.)
  dgrd  = pai/180.
  rloni = dloni*dgrd
  rloni2= 2.*rloni
  rlati = dlati*dgrd
  rlati2= 2.*rlati
  oum2 = 2.*2.*pai/tday
  
!contains
! You can input zero into flux where the flux is undefined.      
      do ilat=1,mlat
         do ilon=1,mlon
            tnf(:,ilon,ilat)=0.
            v(ilon,ilat)=0.
            u(ilon,ilat)=0.
         enddo
      enddo

! dpsi/dx dpis/dy.
      do ilat=2,mlat-1

         ilatn=ilat+1
         ilats=ilat-1

         do ilon=1,mlon

            ilone=ilon+1
            if(ilone.gt.mlon) ilone=1

            ilonw=ilon-1
            if(ilonw.lt.1) ilonw=mlon

            psiame = psiam(ilone,ilat)
            psiamw = psiam(ilonw,ilat)
            psiamn = psiam(ilon,ilatn)
            psiams = psiam(ilon,ilats)

            v(ilon,ilat) = (psiame-psiamw)/(rloni2)
            u(ilon,ilat) = (psiamn-psiams)/(rlati2)
         enddo
      enddo
      
      do ilat=3,mlat-2

! If the latitude direction of input data is north to south, change the following lines !!"
!  and definition of neqlats and neqlatn

         if(ilat .lt. neqlats .and. ilat .gt.neqlatn) cycle

         ilatn=ilat+1
         ilats=ilat-1

         rlat=dgrd*(90.-real((ilat-1))*dlati)
         coslat=cos(rlat)

         do ilon=1,mlon

! If westerly wind of the basic state is smaller than ubmin or easterly, 
! flux is unavailable.

            ub=um(ilon,ilat)
            if(ub.lt.ubmin) cycle

            vb=vm(ilon,ilat)
            wspeed = sqrt(ub*ub+vb*vb)

            ilone=ilon+1
            if(ilone.gt.mlon) ilone=1

            ilonw=ilon-1
            if(ilonw.lt.1) ilonw=mlon

! v equals dpsi/dlamda u equals dpsi/dphi
            ub  =um(ilon,ilat)
            vb  =vm(ilon,ilat)

            vam =v(ilon,ilat)
            uam =u(ilon,ilat)

            psia=psiam(ilon,ilat)

            uamn=u(ilon,ilatn)
            uams=u(ilon,ilats)

            uame=u(ilone,ilat)
            uamw=u(ilonw,ilat)

            vame=v(ilone,ilat)
            vamw=v(ilonw,ilat)

            dudx = (uame-uamw)/(rloni2)

            dudy = (uamn-uams)/(rlati2)

            dvdx = (vame-vamw)/(rloni2)

            termxU=vam*vam-psia*dvdx
            termxV=uam*vam-psia*dudx
            termyU=termxV
            termyV=uam*uam-psia*dudy

            fx=ub*termxU/(coslat*coslat)+vb*termxV/coslat
            fy=ub*termyU/coslat+vb*termyV

            tnf(1,ilon,ilat)=prem*coslat/(2.*wspeed*erd*erd)*fx
            tnf(2,ilon,ilat)=prem*coslat/(2.*wspeed*erd*erd)*fy
         enddo
      enddo

      return
end subroutine tnf_xy