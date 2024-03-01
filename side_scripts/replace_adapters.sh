#!/bin/bash

while getopts o: flag
do
    case "${flag}" in
        o) organism=${OPTARG};;
    esac
done

cat $organism/sample_ids.txt | while read line;

do

echo $line
sed -i 's/TruSeq3-PE-2.fa/NexteraPE-PE.fa/' $organism/scripts/trimmomatic$line.sh

done
