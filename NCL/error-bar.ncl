;---------------------------------------------------------------------------------
; This procedure can draw error bar.
; new_index: new sumbol.
; COLOR:    error bar color.
; Lthick:   error bar thickness.
; MarkSiz:  two sides marker size.
; Dpattern: line dashpattern.
;--------------------------------------------------------------------------------

undef("ErrorBar")
procedure ErrorBar(wks1,plotebar,xx,yy,yup,ydown)
local wks1,xx,yy,yup,ydown,up1,down1,polyres,polyres2,\
dum1,plotebar
begin
  mstring =       "-"
  fontnum =       19
  xoffset =       0.0
  yoffset =       0.0
  aspect  =       1E-10
  size    =       1.2
  angle   =       0.0

  new_index = NhlNewMarker(wks, mstring, fontnum, xoffset, yoffset,\
   aspect, size, angle)

  COLOR    = ""
  LThick   = 2
  MarkSiz  = 0.03
  Dpattern = 0
  polyres                       = True
  polyres@gsLineColor           = COLOR
  polyres@gsLineDashPattern     = Dpattern
  polyres@gsLineThicknessF      = LThick
  polyres@tfPolyDrawOrder       = "PostDraw"

  polyres2                      = polyres
  polyres2@gsMarkerIndex        = new_index
  polyres2@gsMarkerSizeF        = Marksiz

  dum1   = gsn_add_polyline(wks1,plotebar,xx1,yy1,polyres)
  up1    = gsn_add_polymarker(wks1,plotebar,xx1,yup,polyres2)
  down1  = gsn_add_polymarker(wks1,plotebar,xx1,ydown,polyres2)

end