#!/bin/bash
models='uu ceh jrc cnr metfr eth csiro unikassel ecmwf'
models='ecmwf'
vars='SWE'
for mod in $models;do
for var in $vars;do
#echo ftp://e2o_guest:oowee3WeifuY1aeb@wci.earth2observe.eu/data/primary/public/$mod/wrr2/*${var}_2000-2014.nc
mkdir -p /scratch/hydrodl/data/EartH2O/ver2/$mod/
wget --user=e2o_guest --password='"oowee3WeifuY1aeb"' ftp://wci.earth2observe.eu/data/primary/public/$mod/wrr2/*${var}_2000-2014.nc -P /scratch/hydrodl/data/EartH2O/ver2/$mod/
done
done
