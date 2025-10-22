;---------------------------------------------------------
;; use effective method to create netcdf file.
; variables info:
; x: required data (time,lat,lon)
; LON: longitude
; LAT: latitude
; TIME: time.
; LEVEL: pressure level.
; DIR: outdir.
; VAR: variables name.
; default: x(time|:, level|:, lat|:, lon|:)
;---------------------------------------------------------

undef("create_nc_file")

procedure create_nc_file(x:float,TIME,LEVEL,LAT,LON,DIR:string,VAR:string)
local x,LON,LAT,TIME,DIR,VAR,fout,LEVEL
begin
    system("/bin/rm -f "+DIR)

  fout = addfile(DIR,"c")

  dimNames = (/"time", "level", "lat", "lon"/)  
  dimSizes = (/ -1   ,  dimsizes(LEVEL), dimsizes(LAT),  dimsizes(LON)/) 
  dimUnlim = (/ True , False, False, False/)   
  filedimdef(fout,dimNames,dimSizes,dimUnlim)

  filevardef(fout, "time" ,typeof(TIME),getvardims(TIME))  
  filevardef(fout, "level" ,typeof(LEVEL),getvardims(LEVEL))                          
  filevardef(fout, "lat"  ,typeof(LAT),getvardims(LAT))                          
  filevardef(fout, "lon"  ,typeof(LON),getvardims(LON))                          
  filevardef(fout, VAR    ,typeof(x)  ,getvardims(x))    
                                                                                             
  filevarattdef(fout,"time" ,TIME)   
  filevarattdef(fout,"level" ,LEVEL)                 
  filevarattdef(fout,"lat"  ,LAT)                   
  filevarattdef(fout,"lon"  ,LON)                     
  filevarattdef(fout, VAR   ,x)   

  setfileoption(fout,"DefineMode",False)

  fout->time          = (/TIME/)  
  fout->level         = (/LEVEL/)    
  fout->lat           = (/LAT/)
  fout->lon           = (/LON/) 
  fout->$VAR$         = (/x/)

end