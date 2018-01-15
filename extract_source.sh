#! /bin/bash
# Bash script to reprocess the data

for obs in $(find "$PWD/data" -type d -name '*' -mindepth 1 -maxdepth 1)
do
	obs_id=$(basename "$obs")
	chandra_repro "$obs" "repro_$obs_id" clob+
	cd "repro_$obs_id"
	echo $obs_id
	while [ ${#obs_id} -lt 5 ]
	do
		obs_id="0$obs_id"
		echo $obs_id
	done
	dmcopy infile="acisf"$obs_id"_repro_evt2.fits[sky=region(../source_ref_fk5.reg)]" outfile="source_ref_evt2.fits" clob+
	dmcopy infile="source_ref_evt2.fits[bin x=::0.125, y=::0.125]" outfile="sourceref_rebinned_evt2.fits" clob+
	dmcopy infile="source_ref_evt2.fits[bin x=::0.5, y=::0.5]" outfile="sourceref_halfbin_evt2.fits" clob+
	aconvolve infile="sourceref_rebinned_evt2.fits" outfile="sourceref_conv.fits" kernelspec="lib:gauss(2, 5, 1, 0.2158, 0.2158)" method=fft clob+
	cd ..
done
