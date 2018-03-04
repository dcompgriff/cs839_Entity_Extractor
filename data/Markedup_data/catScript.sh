#!/bin/bash

#This script accepts three arguments.
#$1 is the first file number.
#$2 is the last file number.
#$3 is the string to search for. NOTE: use "<\[>" for tags, and the script
#is only configured for the current number of files in the directory.
#Uncomment the line echo $myList to see the true range of files being 
#checked.

#Example usage:
#    Daniels-MacBook-Pro:Markedup data dcompgriff$ ./catScript.sh 200 300 "<\[>"
#    The string <\[> occurs: 543 times.

N=$1;
M=$2;
myStr=$3;
((low=$M+3));
((high=$M-$N+1));

myList=$(ls | sort -g | head -n $low | tail -n $high);
#echo $low;
#echo $high;
echo $myList;
myCount=$(cat $myList | grep -c $myStr);
echo The string $myStr occurs: $myCount times.;





