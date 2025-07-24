#! /bin/bash
# dir: data directory.
# xf: x data file name.
# yf: y data file name.
# ouname: ouput file name.
# xvar: x variable name.
# yvar: y variable name.

i=$1
export dir=""
export pythondir=""

if [ i -eq 2 ]; then

	export xf="shum2m.JJA.nc"
	export yf="t2m.JJA.nc"
	export xvar="shum"
	export yvar="air"
	export ouname="output.nc"

	${pythondir}  Main_Bivariate_Analysis.py
	unset xf;unset yf;unset xvar;unset yvar;unset ouname
else
	export xf="shum2m.JJA.nc"
	export xvar="shum"
	export ouname="output.nc"

	${pythondir}  Main_Univariate_Analysis.py

	unset xf;unset xvar
fi

unset dir;unset pythondir



