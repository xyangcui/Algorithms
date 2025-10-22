;---------------------------------------------------------------------------
; This scrpit intends to caculate the Rossby wave ray tracing.
; Parameters this script needs are listed as follow.
; um: background zonal wind (nlat,nlon)
; umz: um in Mecartor corodinates.
; betam: absolute vorticity y gradient. calculate with Karoly (1983).
; x0 y0: primary point.
; K:  zonal wave number.
; Ks: stational wave number.
; dt: time steps. unit:s.
; iday: integtated day.
; itime: integerated  time 
; cx: phase velocity default is 0.
; ug vg: group velocity 
; Information
; author: Cui xiangyang 
;---------------------------------------------------------------------------

function raytracing(um,x0,y0,dt,iday,K,cx)
local a,pi,omega,itime,dir,lat,nlat,phirad,phi,YY,\
umz,b,d,c,dfdy,tm2,d2udy2,beta_m,k,l,y
begin 
;---------------------------------------------------------------------------
;some constants
a     = 6371000.              ;Earth's radius (unit:m)
pi    = atan(1.0)*4.          ;pi
omega = 2.*pi/(24.*60.*60.)   ;Earth's angular velociy (unit: s-1)               
itime = iday*4.*6.*60.*60.                            
dir   = -1.                   ;direction: 1. is poleward; -1. is equatward.
;---------------------------------------------------------------------------
lat      = um&lat
nlat     = dimsizes(lat)
phirad   = lat*pi/180.   ;(90 - -90)
phi  = conform_dims(dimsizes(um), phirad, 0)
;;calculate beta terms in Mercator
YY   = phi
do i = 0, nlat-1
    YY(i,:)   = a*log((1.+sin(phi(i,:)))/cos(phi(i,:))) 
end do
;---------------------------------------------------------------------------
; Mercator coordinates         
umz  = um/cos(phi)                         ; uwind in Mercator.
umz(0,:) = 0.
umz(nlat-1,:) = 0.
copy_VarCoords(um, umz)
b = (1.+cos(2*phi))/2.                     ; cos^2
d = new(dimsizes(b),float)                 ; 1/cos^2
do i=1,nlat-2
    d(i,:) = 1/b(i,:)
end do
c       = umz*b
dfdy    = c
tm2     = c
d2udy2  = c
;1st term: df/dy
dfdy = 2*omega*b/a
;2nd term: d2u/dy2
tm2  = center_finite_diff_n(c, YY, False, 0, 0)
tm2 := tm2*d
d2udy2 = center_finite_diff_n(tm2, YY, False, 0, 0)
;betam = df/dy - d2u/dy2
beta_m = dfdy - d2udy2
copy_VarCoords(um, beta_m)
delete([/dfdy,tm2,d2udy2/])
beta_m := beta_m(3:nlat-4,:)

umz    := dim_avg_n_Wrap(umz,1) 
beta_m := dim_avg_n_Wrap(beta_m,1)
YY     := YY(:,1)
;---------------------------------------------------------------------------
;caculate cgx.
function UG(um1,betam,ku,lu,ks)
local um1,betam,ku,lu,ks
begin
ug = (um1 + betam*(ku^2 - lu^2)/(ks^2))*180./(a*pi)
return(ug)
end

;caculate cgy.
function VG(um1,betam,kv,lv,ks,tc)
local um1,betam,kv,lv,ks,tc
begin
vg = (2.*betam*kv*lv/(ks^2))*tc*180./(a*pi)
return(vg)
end

;caculate x and y.
function cacls(z1,wnd,delta)
local z1,wnd,delta
begin
z = z1 + delta*wnd
return(z)
end
;---------------------------------------------------------------------------
;Main code.
icount    = floattointeger(itime/dt)
latlon    = new((/2,icount+1/),float)
latlon    = 0.
latlon(1,0)  = x0
latlon(0,0)  = y0
cgx   = (/0.,0.,0.,0./)
cgy   = cgx
k     = K/(a*cos(y0*pi/180.))
K2    = k^2
tl      = 1.
YYu     = YY(1:nlat-2)
YYb     = YY(3:nlat-4)
umz     := umz(1:nlat-2)
;---------------------------------------------------------------------------
ftsetp("sig",0.0)
do i = 0 , icount-1               
    ;K1 term
    y = a*log((1.+sin(y0*pi/180.))/cos(y0*pi/180.))  
    u_y     = ftcurv(YYu(::-1), umz(::-1), y)     
    beta_y  = ftcurv(YYb(::-1), beta_m(::-1), y)       
    Ks2_y0  = beta_y / u_y
    ;break condition
    if (Ks2_y0.lt.0.) then
        break
    end if
    d12_y0 = Ks2_y0 - K2
    if (d12_y0.lt.0) then
        tl = -1.
    end if
    l   = sqrt(abs(Ks2_y0 - K2))

    cgx(0) = UG(u_y,beta_y,k,l,Ks2_y0)
    cgy(0) = VG(u_y,beta_y,k,l,Ks2_y0,tl)*dir*cos(y0*pi/180.)  ;K1 term
    ;K2 term
    ytmp = cacls(y0,cgy(0),dt/2.)   ; x1 y1

    y = a*log((1.+sin(ytmp*pi/180.))/cos(ytmp*pi/180.))  
    u_y     = ftcurv(YYu(::-1), umz(::-1), y)     
    beta_y  = ftcurv(YYb(::-1), beta_m(::-1), y) 

    Ks2_y0 = beta_y / u_y
    ;break condition
    if (Ks2_y0.lt.0.) then
        break
    end if
    l   = sqrt(abs(Ks2_y0 - K2))

    cgx(1) = UG(u_y,beta_y,k,l,Ks2_y0)
    cgy(1) = VG(u_y,beta_y,k,l,Ks2_y0,tl)*dir*cos(ytmp*pi/180.) ;K2 term
    ;K3 term.
    ytmp = cacls(y0,cgy(1),dt/2.)  ; x1 nochange, y use K2 to caculate

    y = a*log((1.+sin(ytmp*pi/180.))/cos(ytmp*pi/180.))  
    u_y     = ftcurv(YYu(::-1), umz(::-1), y)     
    beta_y  = ftcurv(YYb(::-1), beta_m(::-1), y)  

    Ks2_y0  = beta_y / u_y
    ;break condition
    if (Ks2_y0.lt.0.) then
        break
    end if
    l   = sqrt(abs(Ks2_y0 - K2))

    cgx(2) = UG(u_y,beta_y,k,Ks2_y0,l)
    cgy(2) = VG(u_y,beta_y,k,l,Ks2_y0,tl)*dir*cos(ytmp*pi/180.)  ;K3 term
    ;K4 term.
    ytmp = cacls(y0,cgy(2),dt)      ;y change for deltachange and K3 value

    y = a*log((1.+sin(ytmp*pi/180.))/cos(ytmp*pi/180.))  
    u_y     = ftcurv(YYu(::-1), umz(::-1), y)     
    beta_y  = ftcurv(YYb(::-1), beta_m(::-1), y) 

    Ks2_y0 = beta_y / u_y
    ;break condition
    if (Ks2_y0.lt.0.) then
        break
    end if
    l   = sqrt(abs(Ks2_y0 - K2))
 
    cgx(3) = UG(u_y,beta_y,k,l,Ks2_y0)
    cgy(3) = VG(u_y,beta_y,k,l,Ks2_y0,tl)*dir*cos(ytmp*pi/180.)  ;K4 term
    ;Great sum
    xt   = x0 + dt*(cgx(0) + 2*cgx(1) + 2*cgx(2) + cgx(3))/6.
    yt   = y0 + dt*(cgy(0) + 2*cgy(1) + 2*cgy(2) + cgy(3))/6.
    ;xt   = x0 + dt*(cgx(0) + cgx(1))/2.
    ;yt   = y0 + dt*(cgy(0) + cgy(1))/2.
    latlon(0,i+1) = yt
    latlon(1,i+1) = xt
    x0  = xt
    y0  = yt
    delete([/xt,yt/])
end do
;---------------------------------------------------------------------------
;create .csv file to retain the values of latitude and longitude.
;csv_filename = "/home/yinqc/script/Ray.csv"
;dq  = str_get_dq()

;fields = (/"lat","lon"/)
;fields = dq + fields + dq	
;header = [/str_join(fields,",")/]

;alist  = [/latlon(0,:),latlon(1,:)/]
;format = "%2.4f,%2.4f"
;system ("rm -f" +" "+ csv_filename)

;write_table(csv_filename, "w", header, "%s")
;write_table(csv_filename, "a", alist, format)
return(latlon)

end
