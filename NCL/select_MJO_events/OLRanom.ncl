
;-------------------------------------------------------
; calculate OLR anomalies by subtracting daily mean 
; and the first 3 harmonics of climatology annual cycle.
; 
;-------------------------------------------------------
datadir = getenv("INPUTDIR")
outdir  = getenv("OUTPUTDIR")
var     = getenv("var")
cbyear  = getenv("cbyear")
ceyear  = getenv("ceyear")
byear   = getenv("byear")
eyear   = getenv("eyear")
;-------------------------------------------------------
undef("iso_filter")
function iso_filter(x,npts,nwa,nwb,sm,cbyear,ceyear)
local time,ymdhms,iyear,imon,iday,ddd,yyyyddd,wts,xClmDay,xClmDay_sm,xAnom,x1,npts,nwa,nwb,x,sm,cbyear,ceyear,byear,eyear
begin
    npts = 201
    time    = x&time
    ymdhms  = cd_calendar(time, 0)
    iyear   = floattoint( ymdhms(:,0) )
    imon    = floattoint( ymdhms(:,1) )   
    iday    = floattoint( ymdhms(:,2) )
    ddd     = day_of_year(iyear, imon, iday) 
    yyyyddd = iyear*1000 + ddd 
    wts     = filwgts_lanczos (npts, 2, 1./nwa,1./nwb , 1.0)
    nd  = ind(iyear.ge.cbyear.and.iyear.le.ceyear)
    xa  = x(nd,:,:)
    yyyyddd1 = yyyyddd(nd)
    if(sm.eq.1) then
         xClmDay    = clmDayTLL(xa, yyyyddd1)
         xClmDay_sm = smthClmDayTLL(xClmDay, 3)
         xAnom      = calcDayAnomTLL(x, yyyyddd, xClmDay_sm)
    else
         xClmDay    = clmDayTLL(xa, yyyyddd1)
         xAnom      = calcDayAnomTLL(x, yyyyddd, xClmDay)
    end if
    x1 = wgt_runave_n_Wrap(xAnom, wts, 0, 0)

    return(x1)
end
;--------------------------------------------------------------------
; Main code.
;--------------------------------------------------------------------
f = addfile(datadir,"r")

time = f->time
TIME = cd_calendar(time, 0)
year = toint(TIME(:,0))
nt   = ind(year.ge.byear.and.year.le.eyear)

olr  = f->$var$(nt,:,:)

olranom = iso_filter(olr,201,70,20,1,cbyear,ceyear)

system ("rm -r "+outdir)

fout = addfile(outdir,"c")
fout ->$var$ = olranom
