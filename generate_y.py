from matplotlib import pyplot as plt
from PIL import Image
import cv2
import numpy as np
import numpy.ma as ma
import sys

def binarylab(labels, width, height):
    x = np.zeros([width,height,21])    
    for i in range(width):
        for j in range(height):
            x[i,j,labels[i][j]]=1
    return x

## Set parameters
width = 256
height = 256
nlabels = 21
voc_loc = "/media/francis/datasets/_AADatasets/VOC2012_orig/"
filename = sys.argv[3]

outfilename = sys.argv[1]+str(sys.argv[2]) #"Ytrain_"
pang_ilan = sys.argv[2]					   # 1
num_get = 100
Y = np.zeros((num_get,width,height,nlabels));
overallcount = 1
count = 1

## Find location of data
print("PROCESSING: %s" % (filename))
with open (voc_loc+filename, "r") as myfile:
	in_list = myfile.readlines()
	in_list = [item.strip() for item in in_list]
num_in = len(in_list)

## Read and resize inputs and segmeneteds (TRAINING)
for cnt in range(num_in):
	if(overallcount>(num_get*(int(pang_ilan)-1))):
		curr_file = voc_loc+"SegmentationClass_back/"+in_list[cnt]+".png_e"
		print(curr_file)
		curr_image = binarylab(cv2.resize(np.array(Image.open(curr_file)), (width, height), interpolation = cv2.INTER_NEAREST), width, height)
		print(np.unique(curr_image))

		# Compilation
		Y[count-1,:,:,:] = curr_image
		print("----------- %s/%s -----------" % (pang_ilan,filename))
		print("Processed: %d of %d" % (count,num_get))
		count = count+1

	overallcount = overallcount+1
	if(count>num_get):
		break
	
print("DONE.")
Y = np.array(Y)
Y = np.reshape(Y,(num_get,width*height,nlabels))

# Saving
print(Y.shape)
print("SAVING AS .NPZ FILE: %s" % (outfilename))
np.save(outfilename, Y)