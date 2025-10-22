#! /bin/bash
# Input data is daily OLR. 
# Output contains two parts. An nc file contains olr anomalies and a csv file contains MJO central date.
# two ncl files. one calculate anomaly and the other counter MJO events.
# Information
# author: cui xiangyang time: ????????? release: v1
#-------------------------------------------------------------------------------------------------------
# declare some environmental variablies.
#-------------------------------------------------------------------------------------------------------
export INPUTDIR=""
export OUTPUTDIR=""
export CSVDIR=""
export Calcdir="" 
export selectdir=""

export var="olr"
export lonmin=
export lonmax=
export latmin=
export latmax=
#------------------------------------------------------------------------------------------------------
# calculate anomalies
#------------------------------------------------------------------------------------------------------

export cbyear=
export ceyear=
export byear=
export eyear=

ncl ${Calcdir}/OLRanom.ncl
#------------------------------------------------------------------------------------------------------
# Main code.
#------------------------------------------------------------------------------------------------------

ncl ${selectdir}/select_MJO.ncl


