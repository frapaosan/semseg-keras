from matplotlib import pyplot as plt
from PIL import Image
import cv2
import numpy as np
import sys

def normalized(rgb):
    #return rgb/255.0
    norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)

    b=rgb[:,:,0]
    g=rgb[:,:,1]
    r=rgb[:,:,2]

    norm[:,:,0]=cv2.equalizeHist(b)
    norm[:,:,1]=cv2.equalizeHist(g)
    norm[:,:,2]=cv2.equalizeHist(r)

    return norm

## Set parameters
width = 256
height = 256
voc_loc = "/media/francis/datasets/_AADatasets/VOC2012_orig/"
filename = sys.argv[3]

outfilename = sys.argv[1]+str(sys.argv[2]) #"Xtrain_"
pang_ilan = sys.argv[2]					   # 1
num_get = 100
X = []
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
		curr_file = voc_loc+"JPEGImages/"+in_list[cnt]+".jpg"
		print(curr_file)
		curr_image = normalized(cv2.resize(plt.imread(curr_file), (width, height)))
		#plt.imshow(curr_image)
		#plt.show()

		# Compilation
		X.append(curr_image)
		print("----------- %s -----------" % pang_ilan)
		print("Processed: %d of %d" % (count,num_get))
		count = count+1

	overallcount = overallcount+1
	if(count>num_get):
		break
		
print("DONE.")
X = np.array(X)

# Saving
X = np.array(X)
print(X.shape)
print("SAVING AS .NPZ FILE: %s" % (outfilename))
np.save(outfilename, X)