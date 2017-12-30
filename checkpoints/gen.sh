#!/bin/sh
# Generate X .npy files
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 
do
   python generate_x.py Xtrain_ $i train.txt
done
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
do
   python generate_x.py Xval_ $i val.txt
done
# Generate Y .npy files
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 
do
   python generate_y.py Ytrain_ $i train.txt
done
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
do
   python generate_y.py Yval_ $i val.txt
done