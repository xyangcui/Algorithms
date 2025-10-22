;************************************************* 
; traj_1.ncl
;
; Concepts illustrated:
;   - Plotting trajectories
;*************************************************
;
; These files are loaded by default in NCL V6.2.0 and newer
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
;**************************************************
begin
;*************************************
; read in data
;************************************
; note this trajectory file contains data in the form of
; 9 variables x 131 timesteps x 100 trajectories
   f        = asciiread("/home/yinqc/script/Ray.csv", -1, "string")
   delim = ","
   x = str_get_field(f,2,delim)
   y = str_get_field(f,1,delim)

;********************************************
   wks = gsn_open_wks("eps","/home/yinqc/script/traj")               ; send graphics to PNG file
   res            = True                          ; map resources
   res@gsnDraw    = False                         ; don't draw
   res@gsnFrame   = False                         ; don't advance frame
   res@gsnMajorLatSpacing = 50.
   res@gsnMajorLonSpacing = 50.
   res@vpWidthF   = 0.80                          ; make map bigger
   res@vpHeightF  = 0.80
   ;res@mpMaxLatF  = 50                           ; select subregion
   ;res@mpMinLatF  = -80
   ;res@mpMinLonF  = -150
   ;res@mpMaxLonF  = 150
   res@mpFillDrawOrder = "PostDraw"
   res@mpFillOn        = False
   res@tiMainString    = "Example of a trajectory plot"  ; title
   res@gsnCenterString = "markers every 4th time step"   ; center string
  

   map = gsn_csm_map_ce(wks,res)                         ; create map
;*********************************************
; trajectory parameters
;*********************************************
   traj = (/1,10,53,67,80/)          ; choose which trajectories indexes to plot
;*********************************************
; some plot parameters
;*********************************************
   pres                  = True               ; polyline resources
   pres@gsLineThicknessF = 2.0                ; line thickness
   colors= (/"red","blue","dark green","grey","magenta"/) ; line color

   mres  = True                               ; marker resources
   first = True                               ; start of traj resources
;********************************
   c=0                                        ; counter variable               ; loop through chosen traj
      ypt = tofloat(y(1::))             ; extract lat from whole array
      xpt = tofloat(x(1::))+360.             ; extract lon from whole array


      pres@gsLineColor = colors(c)            ; line color for this traj                                ; advance counter
      gsn_polyline(wks,map,xpt,ypt,pres)      ; draw the traj
   draw(map)  
   frame(wks)                                                                  
end

