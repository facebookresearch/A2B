# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved. 

# download data from https://zenodo.org/records/7212795#.Y1XwROxBw-R
shopt -s lastpipe

fld_train="/mnt/home/idgebru/Data/bytedance/train/binaural/"
fld_test="/mnt/home/idgebru/Data/bytedance/test/binaural/"
cnt=0

out="/mnt/home/idgebru/Data/BTPAB/"
rm -rf $out

wavfiles=$(find $fld_train -mindepth 1 -maxdepth 1 -type f  -name "*.wav" -printf '%h\0%d\0%p\n' | sort -t '\0' -n | awk -F '\0' '{print $3}')
for wav in ${wavfiles[@]}; do
	echo "Input:${wav}"
	name=$(echo "$wav" | sed -e 's/\.[^.]*$//')
	echo "Name=$name"
	recid=${name:(-4)}
	echo "RecId:${recid}"
	ambiname="ambisonic"
	amb=${wav//binaural/$ambiname}
	ambiname="AmbiX"
    amb=${amb//Binaural/$ambiname}
	#echo "New:$amb"
    # Output
    outfld=$out/$(printf %04d $cnt)/
    mkdir -p $outfld
	# copy binaural
	outname=$outfld/binaural.wav
	echo $outname
	cp $wav $outname
	# copy ambix
	outname=$outfld/ambisonics.wav
	cp $amb $outname
	cnt=$(( $cnt + 1 ))
	lastcnt="$cnt"
done
echo $lastcnt


wavfiles=$(find $fld_test -mindepth 1 -maxdepth 1 -type f  -name "*.wav" -printf '%h\0%d\0%p\n' | sort -t '\0' -n | awk -F '\0' '{print $3}')

echo "Files: ${cnt}"
#cnt=32
for wav in ${wavfiles[@]}; do
	echo "Input:${wav}"
	name=$(echo "$wav" | sed -e 's/\.[^.]*$//')
	echo "Name=$name"
	recid=${name:(-4)}
	#echo "RecId:${recid}"
	ambiname="ambisonic"
	amb=${wav//binaural/$ambiname}
	ambiname="AmbiX"
    amb=${amb//Binaural/$ambiname} # Replace "Binaural" with "Ambix"
	#echo "New:$amb"
    # Output
    outfld=$out/$(printf %04d $cnt)/
    mkdir -p $outfld
	# copy binaural
	outname=$outfld/binaural.wav
	echo "Output: ${outname}"
	cp $wav $outname
	# copy ambix
	outname=$outfld/ambisonics.wav
	cp $amb $outname
	cnt=$(( $cnt + 1 ))
done
