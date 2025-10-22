;-----------------------------------------------------------------------------
;Procedure to attach a vertical lavelbar to the right of the plot.
;-----------------------------------------------------------------------------

undef("add_labelbar")
procedure add_labelbar(wks,plot,labels,color)
local vph,vpw,nboes,lbres,lbid,amres,annoid,color
begin

    lbLabelFontHei  = 0.02       ;colorbar label fontheight.
    lbLabelOffset   = 0.05       ;label distance between label and bar. larger more distance.
    lbTitleFontHei  = 0.02       ;Title fontheight.
    lbMinorExtent   = 0.2        ;
    lbFillOpacity   = 1.0        ;Opacity color.
    lbOritation     = "Vartical" ;"Horizontal"/"Vertical".
    amParral        = 0.55       ;X position.
    amOrtho         = 0.0        ;Y position.

    getvalues plot
      "vpHeightF" : vph
      "vpWidthF"  : vpw
    end getvalues

    lbres = True

    lbres@vpWidthF           = 0.15*vpw
    lbres@vpHeightF          = 0.95*vph

    lbres@lbAutoManage       = False
    lbres@lbFillColors       = color
    lbres@lbMonoFillPattern  = True

    lbres@lbLabelFontHeightF = lbLabelFontHei
    lbres@lbLabelOffsetF     = lbLabelOffset
    lbres@lbLabelAlignment   = "InteriorEdges"

    lbres@lbTitleOn          = False
    lbres@lbTitleFontHeightF = lbTitleFOntHei
    lbres@lbTitleDirection   = "Across"
    lbres@lbTitlePosition    = "Right"
    lbres@lbTitleString      = labels 

    lbres@lbPerimOn          = False
    lbres@lbBoxLinesOn       = True
    lbres@lbBoxMinorExtentF  = lbMinorExtent
    lbres@lbFillOpacityF     = lbFillOpacity
    lbres@lbOrientation      = lbOritation 
    
    lbid = gsn_create_labelbar(wks,dimsizes(labels),labels,lbres)

; attach labelbar to the plot.
    amres = True
    ;amres@amJust           = "CenterLeft"
    amres@amParallelPosF   = amParral
    amres@amOrthogonalPosF = amOrtho
    annoid = gsn_add_annotation(plot,lbid,amres)

end 