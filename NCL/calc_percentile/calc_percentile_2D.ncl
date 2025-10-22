;------------------------------------------------------
; procedure to calculate nth percentile for 2D martrix.
; th_percentile(x,per)
; x: variable like T(ntime,nlat,nlon)
; per: nth percentile.
; author: cui xiangyang 
;------------------------------------------------------

;------------------------------------------------------
; function nth percentile.
;------------------------------------------------------
undef("percent_to_value")
function percent_to_value(i_data,i_per)
local i_value,igood,floatInd,floorInd,ceilInd,retVal
begin
  
  if ( any(ismissing(i_data)) ) then
    igood   = ind(.not.ismissing(i_data))
    i_value = i_data(igood)
  else
    i_value = i_data
  end if
  ; sort!!!!!
  qsort(i_value)

  floatInd = i_per*.01*dimsizes(i_value) - 0.5
  floorInd = toint( floor(floatInd) )
  floorInd = where( floorInd.lt.0,0,floorInd )
  ceilInd  = toint( ceil(floatInd) )
  ceilInd  = where( ceilInd.ge.dimsizes(i_value), \
                   dimsizes(i_value)-1,ceilInd)

  if( ceilInd.eq.floorInd) then
    retVal = i_value(floorInd)
  else
    retVal = i_value(floorInd)*(ceilInd-floatInd) \
    + i_value(ceilInd)*(floatInd-floorInd)
  end if

  return(retVal)
  delete([/i_data,i_value,igood,floatInd,floorInd,ceilInd,retVal/])
end

;------------------------------------------------------
; function stats values 5th percentile
;------------------------------------------------------
undef("th_percentile")
function th_percentile(x[*][*][*]:float,per)
local x,y,th,nlat1,nlon1
begin
  nlat1 = dimsizes(x&lat)
  nlon1 = dimsizes(x&lon)
  th = new((/nlat1,nlon1/),float)
  th = 0.

  do i = 0, nlat1-1
    do j = 0, nlon1-1
        th(i,j) = percent_to_value(x(:,i,j),per)
    end do
  end do

  return(th)
  delete([/x,y,th,nlat1,nlon1/])

end 
