#!/bin/bash
models='uu ceh jrc cnr metfr eth csiro unikassel ecmwf'
models='ecmwf'
for mod in $models;do
mkdir -p /Net/Groups/BGI/people/skoirala/Data/EartH2O/ver2/$mod
wget ftp://e2o_guest:oowee3WeifuY1aeb@wci.earth2observe.eu/data/primary/public/$mod/wrr2/*.nc -P /Net/Groups/BGI/people/skoirala/Data/EartH2O/ver2/$mod/
done
