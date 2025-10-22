;-------------------------------------------------------------------
; This function is used to filter.
; wts: weight for Lanczos filter.
; xClmDay: climatology
; xClmDay_sm: smoothed climatology, subtracting first 3 harmonics.
; xAnom: daily anomalies.
; x1: filtered result.
; npts: notes of lanczos filter. 
; nwa: larger value.  
; nwb: smaller value. 
; sm: sm=1, smoothed; sm!=1, nonsmoothed.
;-------------------------------------------------------------------
function iso_filter(x,npts,nwa,nwb,sm)
local time,ymdhms,iyear,imon,iday,ddd,yyyyddd,wts,xClmDay,xClmDay_sm,xAnom,x1,npts,nwa,nwb,x,sm
begin
    time    = x&time
    ymdhms  = cd_calendar(time, 0)
    iyear   = floattoint( ymdhms(:,0) )
    imon    = floattoint( ymdhms(:,1) )   
    iday    = floattoint( ymdhms(:,2) )
    ddd     = day_of_year(iyear, imon, iday) 
    yyyyddd = iyear*1000 + ddd 
    wts     = filwgts_lanczos (npts, 2, 1./nwa,1./nwb , 1.0)
    if(sm.eq.1) then
         xClmDay    = clmDayTLL(x, yyyyddd)
         xClmDay_sm = smthClmDayTLL(xClmDay, 3)
         xAnom      = calcDayAnomTLL(x, yyyyddd, xClmDay_sm)
    else
         xClmDay    = clmDayTLL(x, yyyyddd)
         xAnom      = calcDayAnomTLL(x, yyyyddd, xClmDay)
    end if
    x1 = wgt_runave_n_Wrap(xAnom, wts, 0, 0)

    return(x1)
end
