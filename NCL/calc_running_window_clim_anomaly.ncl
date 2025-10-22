;----------------------------------------------------------
; calculate t2m anomalies in NA. 
; using a 30-year running window, function of detrending. 
; Information
; author: cui xiangyang time: 2023-10-29 release:v1
; t2m: variable
; t2m_anom: Final anomalies.
;----------------------------------------------------------
datadir = "/home/sunming/data5/cuixy/data/MJO_pan-Atlantic/ERA5_2m_temperature/"
inname  = "ERA5_2m_temperature_1979-2022.nc"
ouname  = "ERA5_2m_temperature_anom_1979-2022.nc"

var = "t2m"
;----------------------------------------------------------
; Function: single climatology
;----------------------------------------------------------
undef("single_clim")
function single_clim(nt, x,yr,mo,da, ymS, ymL)
local nt,x,ddd,yyyyddd,xClimDay,xAnom,year1,mon1,day1,xtmp,Xtmp
begin
  xtmp     = x(nt,:,:)
  year1    = yr(nt)
  mon1     = mo(nt)
  day1     = da(nt)
  ddd      = day_of_year(year1, mon1, day1)
  yyyyddd  = year1*1000 + ddd
  xClimDay = clmDayTLL(xtmp, yyyyddd)

  nd = ind(yr.ge.ymS.and.yr.le.ymL)
  year2     = yr(nd)
  mon2      = mo(nd)
  day2      = da(nd)
  ddd2      = day_of_year(year2, mon2, day2)
  yyyyddd2  = year2*1000 + ddd2
  Xtmp  = x(nd,:,:)
  xAnom = calcDayAnomTLL(Xtmp, yyyyddd2, xClimDay)

  return(xAnom)

end
;----------------------------------------------------------
; Main code
;----------------------------------------------------------

f = addfile(datadir+inname,"r")

time = f->time
TIME = cd_calendar(time, 0)
year = toint(TIME(:,0))
mon  = toint(TIME(:,1))
day  = toint(TIME(:,2))

t2m = f->$var$

;--5-day running mean
t2m := runave_n_Wrap(t2m, 5, -1, 0)

;--New martrix contains anomalies.
nt       = ind(mon.ge.11.and.mon.le.12)
year(nt) = year(nt) + 1
t2m_anom = t2m

;--period 1. (1979-1994) (1979-2009)
ymStrt1 = year(0)
ymLast1 = ymStrt1 + 15
nt11      = ind(year.ge.ymStrt1.and.year.le.(ymStrt1+29))    ;clim
nt12      = ind(year.ge.ymStrt1.and.year.le.ymLast1)
t2mAnom   = single_clim(nt11,t2m,year,mon,day,ymStrt1,ymLast1)

t2m_anom(nt12,:,:) = t2mAnom
copy_VarAtts(t2mAnom, t2m_anom)
delete([/t2mAnom/])

;--period 2. (2006-2021)
ymLast2 = year(dimsizes(year)-1)
ymStrt2 = ymLast2 - 15
nt21      = ind(year.ge.(ymLast2-29).and.year.le.ymLast2)    ;clim
nt22      = ind(year.ge.ymStrt2.and.year.le.ymLast2)    
t2mAnom   = single_clim(nt21,t2m,year,mon,day,ymStrt2,ymLast2)

t2m_anom(nt22,:,:) = t2mAnom
delete([/t2mAnom/])

;--period 3. (1995-2007)
ymStrt = ymStrt1 + 1
ymLast = ymStrt + 29
do i = ymLast1+1,ymStrt-1
    ymStrt  = ymStrt + 1
    ymLast  = ymLast + 1

    nt31 = ind(year.ge.ymStrt.and.year.le.ymLast) ;clim
    t2mAnom  = single_clim(nt31,t2m,year,mon,day,i,i)
    
    nt32 = ind(year.eq.i)
    t2m_anom(nt32,:,:) = t2mAnom
    delete([/t2mAnom,nt31,nt32/])
end do

copy_VarCoords(t2m, t2m_anom)

system("rm -f "+datadir+ouname)

f1  = addfile(datadir+ouname,"c")

f1->$var$ = t2m_anom
