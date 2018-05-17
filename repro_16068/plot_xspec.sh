#!/bin/bash
#Program to plot the xspec figures in python
xspec_file=$1
source ~/.bash_profile
source ~/.bashrc
source ~/.bash_aliases
heasoft
xspec - get_xspecdata.xcm $xspec_file
