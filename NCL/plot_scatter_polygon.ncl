;-----------------------------------------------------------------------------
; Function to draw polygons in a viewpoint space.
; wks: workstation
; plot: plot variable
; values: X-Y 
; ranges: label
; color: colorbar
;-----------------------------------------------------------------------------
undef("add_polygon")
function add_polygon(wks:graphic,plot:graphic,values[*][*],ranges,color)
local plot,values,xmin,xmax,ymin,ymax,nx,ny,fil_color
begin
    dims    = dimsizes(values)
    ny      = dims(1)
    nx      = dims(0)
;loop throw and draw polygons.
    getvalues plot
      "trXMinF" : xmin
      "trXMaxF" : xmax
      "trYMinF" : ymin
      "trYMaxF" : ymax
    end getvalues

    xsize  = 1.
    ysize  = 1.

    gnres                 = True
    gnres@tfPolyDrawOrder = "PostDraw"  
; First row.
    j = 0
    ypos = ymin
    do i=1,nx-2
        fil_color = get_color_rgba(color, ranges, values(i,j))
        gnres@gsFillColor = fil_color;fil_color(i,j,:)
        xpos = xmin + (xsize/2. * (2*i-1))
        xbox = (/xpos,xpos+xsize,xpos+xsize,xpos,xpos/)
        ybox = (/ypos,ypos,ypos+ysize/2.,ypos+ysize/2.,ypos/)    
        str = unique_string("fill_lft")     
        plot@$str$ = gsn_add_polygon(wks,plot,xbox,ybox,gnres)   
    end do
; Left
    i = 0
    fil_color = get_color_rgba(color, ranges, values(i,j))
    gnres@gsFillColor = fil_color;fil_color(i,j,:)
    ;gnres@gsFillColor = fil_color(i,j,:)
    xpos = xmin 
    xbox = (/xpos,xpos+xsize/2.,xpos+xsize/2.,xpos,xpos/)
    ybox = (/ypos,ypos,ypos+ysize/2.,ypos+ysize/2.,ypos/)
    str = unique_string("fill_lft")     
    plot@$str$ = gsn_add_polygon(wks,plot,xbox,ybox,gnres)
; The rightmost.
    i = nx-1
    fil_color = get_color_rgba(color, ranges, values(i,j))
    gnres@gsFillColor = fil_color;fil_color(i,j,:)
    ;gnres@gsFillColor = fil_color(i,j,:)
    xpos = xmin + (xsize/2. * (2*i-1))
    xbox = (/xpos,xmax,xmax,xpos,xpos/)
    ybox = (/ypos,ypos,ypos+ysize/2.,ypos+ysize/2.,ypos/)    
    str = unique_string("fill_lft")     
    plot@$str$ = gsn_add_polygon(wks,plot,xbox,ybox,gnres)

; Top
    j = ny - 1
    ypos = ymax - ysize/2.
    do i=1,nx-2
        fil_color = get_color_rgba(color, ranges, values(i,j))
        gnres@gsFillColor = fil_color;fil_color(i,j,:)
        ;gnres@gsFillColor = fil_color(i,j,:)
        xpos = xmin + (xsize/2. * (2*i-1))
        xbox = (/xpos,xpos+xsize,xpos+xsize,xpos,xpos/)
        ybox = (/ypos,ypos,ymax,ymax,ypos/)    
        str = unique_string("fill_lft")     
        plot@$str$ = gsn_add_polygon(wks,plot,xbox,ybox,gnres)    
    end do
; Left
    i = 0
    fil_color = get_color_rgba(color, ranges, values(i,j))
    gnres@gsFillColor = fil_color;fil_color(i,j,:)
    ;gnres@gsFillColor = fil_color(i,j,:)
    xpos = xmin 
    xbox = (/xpos,xpos+xsize/2.,xpos+xsize/2.,xpos,xpos/)
    ybox = (/ypos,ypos,ymax,ymax,ypos/)    
    str = unique_string("fill_lft")     
    plot@$str$ = gsn_add_polygon(wks,plot,xbox,ybox,gnres)
; The rightmost.
    i = nx-1
    fil_color = get_color_rgba(color, ranges, values(i,j))
    gnres@gsFillColor = fil_color;fil_color(i,j,:)
    ;gnres@gsFillColor = fil_color(i,j,:)
    xpos = xmin + (xsize/2. * (2*i-1))
    xbox = (/xpos,xmax,xmax,xpos,xpos/)
    ybox = (/ypos,ypos,ymax,ymax,ypos/)    
    str = unique_string("fill_lft")     
    plot@$str$ = gsn_add_polygon(wks,plot,xbox,ybox,gnres)
;Left column
    i = 0
    xpos = xmin
    do j = 1,ny-2
        fil_color = get_color_rgba(color, ranges, values(i,j))
        gnres@gsFillColor = fil_color;fil_color(i,j,:)
        ypos = ymin + (ysize/2.*(2*j-1))
        ;gnres@gsFillColor = fil_color(i,j,:)
        xbox = (/xpos,xpos+xsize/2.,xpos+xsize/2.,xpos,xpos/)
        ybox = (/ypos,ypos,ypos+ysize,ypos+ysize,ypos/)    
        str = unique_string("fill_lft")     
        plot@$str$ = gsn_add_polygon(wks,plot,xbox,ybox,gnres)  
    end do
; Right column
    i = nx-1
    xpos = xmax - xsize/2.
    do j = 1,ny-2
        fil_color = get_color_rgba(color, ranges, values(i,j))
        gnres@gsFillColor = fil_color;fil_color(i,j,:)
        ypos = ymin + (ysize/2.*(2*j-1))
        ;gnres@gsFillColor = fil_color(i,j,:)
        xbox = (/xpos,xmax,xmax,xpos,xpos/)
        ybox = (/ypos,ypos,ypos+ysize,ypos+ysize,ypos/)    
        str = unique_string("fill_lft")     
        plot@$str$ = gsn_add_polygon(wks,plot,xbox,ybox,gnres)  
    end do
;Middle values.
    do j=1,ny-2
        ypos = ymin + (ysize/2.*(2*j-1))
        do i=1,nx-2
            fil_color = get_color_rgba(color, ranges, values(i,j))
            gnres@gsFillColor = fil_color;fil_color(i,j,:)
            ;gnres@gsFillColor = fil_color(i,j,:)
            xpos = xmin + (xsize/2.*(2*i-1))
            xbox = (/xpos,xpos+xsize,xpos+xsize,xpos,xpos/)
            ybox = (/ypos,ypos,ypos+ysize,ypos+ysize,ypos/)    
            str = unique_string("fill_lft")     
            plot@$str$ = gsn_add_polygon(wks,plot,xbox,ybox,gnres)  
        end do
    end do

    return(plot)
end