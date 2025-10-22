#! /bin/bash
############################################################
# Information
# calc streamfunction and xxx.
############################################################

ddir=""
sdir=""

xvar=""
yvar="gph"
ovar="$1"

dirx="${ddir}/"
diry="${ddir}/"
outdir="${ddir}/"

"pythonpath" $sdir

if [ ${ovar} = "phi"]; then

	ncatted -O -a long_name,phi,c,c,streamfunction -a unit,phi,c,c,m2/s ${outdir} 
else
	ncatted -O -a long_name,chi,c,c,velocitypotential -a unit,chi,c,c,m2/s ${outdir}
fi

