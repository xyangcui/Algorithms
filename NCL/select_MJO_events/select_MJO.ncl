;-------------------------------------------------------------------------------------------
; select MJO events by Wang et al.(2019) Sci. Adv. (DJF)
; An MJO envent is selected when box-averaged OLR anomalies over the EIO (10S-10N,75E-95E)
; were below 1SD for five successive days.
; Information
; author:cui xiangyang. time:2023-10-25. release: v1
;-------------------------------------------------------------------------------------------
datadir = getenv("OUTDIR")
lonmin  = getenv("lonmin")
lonmax  = getenv("lonmax")
latmin  = getenv("latmin")
latmax  = getenv("latmax")
outdir  = getenv("CSVDIR")
var     = getenv("var")
pi = atan(1.0)*4.
;-------------------------------------------------------------------------------------------
;function of select DJF.
;-------------------------------------------------------------------------------------------

undef("select_DJF")
function select_DJF(var)
local var,time,TIME,year,mon,revar,vartmp,nt
begin
    time = var&time
    TIME = cd_calendar(time, 0)
    year = toint(TIME(:,0))
    mon  = toint(TIME(:,1))
    day  = toint(TIME(:,2))
;--frist take DJF.
    nt   = ind(mon.eq.1.or.mon.eq.2.or.mon.eq.12.or.mon.eq.11.or.mon.eq.3.or.mon.eq.4)
    vartmp = var(nt,:,:)
;--extract 1979-01,02 and 2022-12
    time := vartmp&time
    TIME := cd_calendar(time, 0)
    year := toint(TIME(:,0))
    mon  := toint(TIME(:,1))
    day  := toint(TIME(:,2))
    nd1   = ind(year.eq.1979.and.mon.eq.11.and.day.eq.1) 
    nd2   = ind(year.eq.2022.and.mon.eq.4.and.day.eq.30) 
    revar = vartmp(nd1:nd2,:,:)

    return (revar)

end

;-------------------------------------------------------------------------------------------
; function select date.
;-------------------------------------------------------------------------------------------
undef("select_date")
function select_date(x[*])
local sd,x,redate,xtmp,xtmp2,nt,time,TIME,label,label2,redate1,year,mon,day,nt,ni
begin
    sd   = stddev(x)
    time = x&time
    TIME = cd_calendar(time, 0)
    year = toint(TIME(:,0))
    mon  = toint(TIME(:,1))
    day  = toint(TIME(:,2))
    ntime  = dimsizes(time)
;--redate martrix.
    redate = new((/ntime,3/),integer)
    xtmp   = sd + x
    copy_VarCoords(x, xtmp)
;--change xtmp time labels for computation.
    label     = toint(ispan(0, ntime-1, 1)) ;times ind.
    xtmp&time = label
    nt     = ind(xtmp.lt.0.)                  ;anomalies' ind lower than 1 sd.
    xtmp2  = xtmp(nt)                         ;anomalies  lower than 1 sd.
    label2 = toint(xtmp2&time)                ;time ind of anomalies lower than 1 sd.
    xdiff  = new(dimsizes(label2)-1,integer)  ;diff between neighboor days.
    xdiff  = 0
;--calculate diff between neighboor days.
    do i = 0,dimsizes(xdiff)-1
        xdiff(i) = label2(i+1) - label2(i)
    end do
;--xdiff consecutive 1.
    i = 0
    j = dimsizes(xdiff)-6
    c = 0
    do while(i.le.j) 
        l  = 0
        if (xdiff(i).eq.1) then
            do k = i+1,j                     ;take all successive days.
                if(xdiff(k).eq.1) then
                    l = l + 1
                else
                    break
                end if
            end do
            ni = sum(xdiff(i:i+l))
            if (ni.ge.4) then
                labels = label2(i:i+l+1)  ;times ind.
                nj     = minind(x(labels))
                ldate  = label(labels(nj))
                redate(c,0) = year(ldate)
                redate(c,1) = mon(ldate)
                redate(c,2) = day(ldate)
                c = c + 1
                i = i + l + 1
                delete([/labels,nj,ldate/])
            else
                i = i + 1
            end if
        else
            i = i + 1       
        end if
    end do
    if (c.eq.0) then
        redate := redate(0:0,:)
    else
        redate := redate(0:c-1,:)
    end if
    return (redate)
    delete([/sd,x,redate,xtmp,xtmp2,nt,time,TIME,label,label2,redate1,year,mon,day,nt,ni/])

end 

;-------------------------------------------------------------------------------------------
; Main code.
;-------------------------------------------------------------------------------------------

f = addfile(datadir,"r")

x    = f->$var$(:,{latmin:latmax},{lonmin:lonmax})

olr  = select_DJF(x)
lat  = olr&lat

time = olr&time
TIME = cd_calendar(time, 0)
year = toint(TIME(:,0))
mon  = toint(TIME(:,1))

clat = cos(lat*pi/180.)
olrb = wgt_areaave_Wrap(olr, clat, 1.0, 0)
delete([/olr,lat,clat/])
;--change times for computation.
nd = ind(mon.eq.1.or.mon.eq.2.or.mon.eq.3.or.mon.eq.4)
reyear     = year
reyear(nd) = reyear(nd) - 1

;--every year selected
date   = new((/150,3/),integer)
begn   = 0
do i = 1979,2021
;    if (i.eq.1986) then
;        continue
;    else
        nt    = ind(reyear.eq.i)
        dates = select_date(olrb(nt))
        n     = dimsizes(dates(:,0))
        endn  = n-1 + begn
        date(begn:endn,:) = dates(:,:)
        begn  = endn + 1
        delete([/dates,nt/])
;    end if
end do

;--remove the _Fill_values.
igood = ind(.not.ismissing(date(:,0)))
date := date(igood,:)
nl  = ind(date(:,1).eq.12.or.date(:,1).eq.1.or.date(:,1).eq.2)
date := date(nl,:)
;--creat csv file to contain year, mon and day.
csv_filename = outdir
dq  = str_get_dq()

fields = (/"year","month","day"/)
fields = dq + fields + dq  
header = [/str_join(fields,",")/]

alist  = [/date(:,0),date(:,1),date(:,2)/]
format = "%d,%d,%d"
system ("rm -f" +" "+ csv_filename)

write_table(csv_filename, "w", header, "%s")
write_table(csv_filename, "a", alist, format)
