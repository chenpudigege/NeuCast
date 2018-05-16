#!/bin/sh
#make cleanall
#make
INPUTDIR="./_elecdat/"
OUTDIR="./_out/"

#----------------------#
echo "----------------------"
echo "  Elec Day Embbeding"
echo "----------------------"
outdir=$OUTDIR"dat_tmp"
dblist=$INPUTDIR"list"
n=1  # data size
d=$1  # dimension
alpha=$2
#----------------------#
rm -rf _out/dat_tmp/dat1
mkdir $outdir
for (( i=1; i<=$n; i++ ))
do
  output=$outdir"/dat"$i"/" 
  mkdir $output
  input=$output"input"
  awk '{if(NR=='$i') print $0}' $dblist > $input
  ./autoplait $d $input $output $alpha
done




