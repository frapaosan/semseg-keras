import os, math
import numpy as np
import random as random
from glob import glob

from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Reshape, Permute
from keras.layers import BatchNormalization, Activation, Input, Dropout, ZeroPadding2D, Lambda
from keras.layers.merge import Concatenate, Add
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json

from keras import backend as K
from keras.backend import tf as ktf
from keras.preprocessing import image

from math import floor, ceil
from resnet50 import ResNet50
from matplotlib import pyplot as plt

from PIL import Image
import sys
import cv2

import keras.models as models
from keras.layers.core import Layer
from keras.layers.convolutional import Convolution2D

# Define Colors
background = [0, 0, 0]				#0
aeroplane = [128, 0, 0]				#1
bicycle = [0, 128, 0]				#2
bird = [128, 128, 0]				#3
boat = [0, 0, 128]					#4
bottle = [128, 0, 128]				#5
bus = [0, 128, 128]					#6
car = [128, 128, 128]				#7
cat = [64, 0, 0]					#8
chair = [192, 0, 0]					#9
cow = [64, 128, 0]					#10
diningtable = [192, 128, 0]			#11
dog = [64, 0, 128]					#12
horse = [192, 0, 128]				#13
motorbike = [64, 128, 128]			#14
person = [192, 128, 128]			#15
pottedplant = [0, 64, 0]			#16
sheep = [128, 64, 0]				#17
sofa = [0, 192, 0]					#18
train = [128, 192, 0]				#19
tvmonitor = [0, 64, 128]			#20
void = [128, 64, 12]				#21
label_colors = np.array([background, aeroplane, bicycle, bird, boat, bottle, bus, car,
	cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor, void])
label_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
			  'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'void']

def apply_colormap(temp):
	# Apply colormap to input image temp
	r = temp.copy()
	g = temp.copy()
	b = temp.copy()
	for l in range(len(label_colors)):
		r[temp==l]=label_colors[l,0]
		g[temp==l]=label_colors[l,1]
		b[temp==l]=label_colors[l,2]

	rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
	rgb[:,:,0] = (r/255.0)#[:,:,0]
	rgb[:,:,1] = (g/255.0)#[:,:,1]
	rgb[:,:,2] = (b/255.0)#[:,:,2]

	return rgb

def getData(folder, ilan_kukunin):
	fileset = sorted(glob(folder+'/*.npy'))
	print(fileset)
	for cnt in range(ilan_kukunin):
		data = np.load(fileset[cnt])
		a = np.array(data)
		del data

		if(cnt==0):
			totData = a
		else:
			totData = np.vstack((totData,a))
		print(totData.shape)
	
	return totData

def BN(name=""):
	return BatchNormalization(momentum=0.95, name=name, epsilon=1e-5)

def residual_conv(prev, level, pad=1, lvl=1, sub_lvl=1, modify_stride=False):
	lvl = str(lvl)
	sub_lvl = str(sub_lvl)
	names = ["conv"+lvl+"_" + sub_lvl + "_1x1_reduce",
			 "conv"+lvl+"_" + sub_lvl + "_1x1_reduce_bn",
			 "conv"+lvl+"_" + sub_lvl + "_3x3",
			 "conv"+lvl+"_" + sub_lvl + "_3x3_bn",
			 "conv"+lvl+"_" + sub_lvl + "_1x1_increase",
			 "conv"+lvl+"_" + sub_lvl + "_1x1_increase_bn"]
	if modify_stride is False:
		prev = Conv2D(64 * level, (1, 1), strides=(1, 1), name=names[0],
					  use_bias=False)(prev)
	elif modify_stride is True:
		prev = Conv2D(64 * level, (1, 1), strides=(2, 2), name=names[0],
					  use_bias=False)(prev)

	prev = BN(name=names[1])(prev)
	prev = Activation('relu')(prev)

	prev = ZeroPadding2D(padding=(pad, pad))(prev)
	prev = Conv2D(64 * level, (3, 3), strides=(1, 1), dilation_rate=pad,
				  name=names[2], use_bias=False)(prev)

	prev = BN(name=names[3])(prev)
	prev = Activation('relu')(prev)
	prev = Conv2D(256 * level, (1, 1), strides=(1, 1), name=names[4],
				  use_bias=False)(prev)
	prev = BN(name=names[5])(prev)
	return prev

def short_convolution_branch(prev, level, lvl=1, sub_lvl=1, modify_stride=False):
	lvl = str(lvl)
	sub_lvl = str(sub_lvl)
	names = ["conv" + lvl+"_" + sub_lvl + "_1x1_proj",
			 "conv" + lvl+"_" + sub_lvl + "_1x1_proj_bn"]

	if modify_stride is False:
		prev = Conv2D(256 * level, (1, 1), strides=(1, 1), name=names[0],
					  use_bias=False)(prev)
	elif modify_stride is True:
		prev = Conv2D(256 * level, (1, 1), strides=(2, 2), name=names[0],
					  use_bias=False)(prev)

	prev = BN(name=names[1])(prev)
	return prev

def empty_branch(prev):
	return prev

def residual_short(prev_layer, level, pad=1, lvl=1, sub_lvl=1, modify_stride=False):
	prev_layer = Activation('relu')(prev_layer)
	block_1 = residual_conv(prev_layer, level,
							pad=pad, lvl=lvl, sub_lvl=sub_lvl,
							modify_stride=modify_stride)

	block_2 = short_convolution_branch(prev_layer, level,
									   lvl=lvl, sub_lvl=sub_lvl,
									   modify_stride=modify_stride)
	added = Add()([block_1, block_2])
	return added

def residual_empty(prev_layer, level, pad=1, lvl=1, sub_lvl=1):
	prev_layer = Activation('relu')(prev_layer)

	block_1 = residual_conv(prev_layer, level, pad=pad,
							lvl=lvl, sub_lvl=sub_lvl)
	block_2 = empty_branch(prev_layer)
	added = Add()([block_1, block_2])
	return added

def ResNet(inp, layers):
	# Names for the first couple layers of model
	names = ["conv1_1_3x3_s2",
			 "conv1_1_3x3_s2_bn",
			 "conv1_2_3x3",
			 "conv1_2_3x3_bn",
			 "conv1_3_3x3",
			 "conv1_3_3x3_bn"]

	# Short branch(only start of network)

	cnv1 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', name=names[0],
				  use_bias=False)(inp)  # "conv1_1_3x3_s2"
	bn1 = BN(name=names[1])(cnv1)  # "conv1_1_3x3_s2/bn"
	relu1 = Activation('relu')(bn1)  # "conv1_1_3x3_s2/relu"

	cnv1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name=names[2],
				  use_bias=False)(relu1)  # "conv1_2_3x3"
	bn1 = BN(name=names[3])(cnv1)  # "conv1_2_3x3/bn"
	relu1 = Activation('relu')(bn1)  # "conv1_2_3x3/relu"

	cnv1 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name=names[4],
				  use_bias=False)(relu1)  # "conv1_3_3x3"
	bn1 = BN(name=names[5])(cnv1)  # "conv1_3_3x3/bn"
	relu1 = Activation('relu')(bn1)  # "conv1_3_3x3/relu"

	res = MaxPooling2D(pool_size=(3, 3), padding='same',
					   strides=(2, 2))(relu1)  # "pool1_3x3_s2"

	# ---Residual layers(body of network)

	"""
	Modify_stride --Used only once in first 3_1 convolutions block.
	changes stride of first convolution from 1 -> 2
	"""

	# 2_1- 2_3
	res = residual_short(res, 1, pad=1, lvl=2, sub_lvl=1)
	for i in range(2):
		res = residual_empty(res, 1, pad=1, lvl=2, sub_lvl=i+2)

	# 3_1 - 3_3
	res = residual_short(res, 2, pad=1, lvl=3, sub_lvl=1, modify_stride=True)
	for i in range(3):
		res = residual_empty(res, 2, pad=1, lvl=3, sub_lvl=i+2)
	if layers is 50:
		# 4_1 - 4_6
		res = residual_short(res, 4, pad=2, lvl=4, sub_lvl=1)
		for i in range(5):
			res = residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i+2)
	elif layers is 101:
		# 4_1 - 4_23
		res = residual_short(res, 4, pad=2, lvl=4, sub_lvl=1)
		for i in range(22):
			res = residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i+2)
	else:
		print("This ResNet is not implemented")

	# 5_1 - 5_3
	res = residual_short(res, 8, pad=4, lvl=5, sub_lvl=1)
	for i in range(2):
		res = residual_empty(res, 8, pad=4, lvl=5, sub_lvl=i+2)

	res = Activation('relu')(res)
	return res

def upsize(x,shape):
	new_height = shape[0]
	new_width = shape[1]
	resized = ktf.image.resize_images(x, [new_height, new_width], align_corners=True)
	return resized

def interp_block(pool_layer, level, feature_map_size, str_lvl=1, ):
	str_lvl = str(str_lvl)

	names = ["pooling"+str_lvl+"_conv",
		"pooling"+str_lvl+"_conv_bn",
		"pooling"+str_lvl+"_ave_pool"]

	kernel = (level, level)
	strides = (level, level)
	pool_layer = AveragePooling2D(kernel, strides=strides, name=names[2])(pool_layer)
	pool_layer = Conv2D(512, (1, 1), strides=(1, 1), name=names[0], use_bias=False)(pool_layer)
	pool_layer = BatchNormalization(momentum=0.95, name=names[1], epsilon=1e-5)(pool_layer)
	pool_layer = Activation('relu')(pool_layer)
	pool_layer = Lambda(upsize, arguments={'shape': feature_map_size})(pool_layer)

	return pool_layer

def pspnet_santelices(input_shape, nb_classes):
	# Implementation from https://github.com/Vladkryvoruchko/PSPNet-Keras-tensorflow
	inp = Input((input_shape[0], input_shape[1], input_shape[2]))

	# Create ResNet50
	#base_model = ResNet50(weights='imagenet', input_shape=input_shape, input_tensor=inp)
	#resnet_model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').input)
	#res = resnet_model.layers[-1].output
	res = ResNet(inp, layers=50)

	# Building Pyramid Pooling Module
	feature_map_size = tuple(int(ceil(input_dim / 8.0)) for input_dim in (input_shape[0],input_shape[1]))
	print(feature_map_size)
	print("PSP module will interpolate to a final feature map size of %s" % (feature_map_size, ))
	
	interp_block1 = interp_block(res, 60, feature_map_size, str_lvl=1)
	interp_block2 = interp_block(res, 30, feature_map_size, str_lvl=2)
	interp_block3 = interp_block(res, 20, feature_map_size, str_lvl=3)
	interp_block6 = interp_block(res, 10, feature_map_size, str_lvl=6)

	# concat all these layers. resulted shape=(1,feature_map_size_x,feature_map_size_y,4096)
	psp = Concatenate()([res,
						interp_block6,
						interp_block3,
						interp_block2,
						interp_block1])

	# Convolution Layer
	x = Conv2D(512, (3, 3), strides=(1, 1), padding="same")(psp)
	x = BatchNormalization(momentum=0.95, epsilon=1e-5)(x)
	x = Activation('relu')(x)
	x = Dropout(0.1)(x)

	#Classification Layer
	x = Conv2D(nb_classes, (1, 1), strides=(1, 1))(x)
	x = Lambda(upsize, arguments={'shape': (input_shape[0], input_shape[1])})(x)
	print(x._keras_shape)
	x = Reshape((nb_classes, input_shape[0]*input_shape[1]))(x)
	x = Permute((2,1))(x)
	
	out = Activation('softmax')(x)

	model = Model(inputs=inp, outputs=out)

	return model

class MaxPoolingWithArgmax2D(Layer):
	def __init__(self, pool_size=(2, 2), strides=(2, 2), padding='same', **kwargs):
		super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
		self.padding = padding
		self.pool_size = pool_size
		self.strides = strides

	def call(self, inputs, **kwargs):
		padding = self.padding
		pool_size = self.pool_size
		strides = self.strides
		if K.backend() == 'tensorflow':
			ksize = [1, pool_size[0], pool_size[1], 1]
			padding = padding.upper()
			strides = [1, strides[0], strides[1], 1]
			output, argmax = K.tf.nn.max_pool_with_argmax(inputs, ksize=ksize, strides=strides, padding=padding)
		else:
			errmsg = '{} backend is not supported for layer {}'.format(K.backend(), type(self).__name__)
			raise NotImplementedError(errmsg)
		argmax = K.cast(argmax, K.floatx())
		return [output, argmax]

	def compute_output_shape(self, input_shape):
		ratio = (1, 2, 2, 1)
		output_shape = [dim // ratio[idx] if dim is not None else None for idx, dim in enumerate(input_shape)]
		output_shape = tuple(output_shape)
		return [output_shape, output_shape]

	def compute_mask(self, inputs, mask=None):
		return 2 * [None]

class MaxUnpooling2D(Layer):
	def __init__(self, size=(2, 2), **kwargs):
		super(MaxUnpooling2D, self).__init__(**kwargs)
		self.size = size

	def call(self, inputs, output_shape=None):
		updates, mask = inputs[0], inputs[1]
		with K.tf.variable_scope(self.name):
			mask = K.cast(mask, 'int32')
			input_shape = K.tf.shape(updates, out_type='int32')
			#  calculation new shape
			if output_shape is None:
				output_shape = (input_shape[0], input_shape[1] * self.size[0], input_shape[2] * self.size[1], input_shape[3])
			self.output_shape1 = output_shape

			# calculation indices for batch, height, width and feature maps
			one_like_mask = K.ones_like(mask, dtype='int32')
			batch_shape = K.concatenate([[input_shape[0]], [1], [1], [1]], axis=0)
			batch_range = K.reshape(K.tf.range(output_shape[0], dtype='int32'), shape=batch_shape)
			b = one_like_mask * batch_range
			y = mask // (output_shape[2] * output_shape[3])
			x = (mask // output_shape[3]) % output_shape[2]
			feature_range = K.tf.range(output_shape[3], dtype='int32')
			f = one_like_mask * feature_range

			# transpose indices & reshape update values to one dimension
			updates_size = K.tf.size(updates)
			indices = K.transpose(K.reshape(K.stack([b, y, x, f]), [4, updates_size]))
			values = K.reshape(updates, [updates_size])
			ret = K.tf.scatter_nd(indices, values, output_shape)
			return ret

	def compute_output_shape(self, input_shape):
		mask_shape = input_shape[1]
		return mask_shape[0], mask_shape[1] * self.size[0], mask_shape[2] * self.size[1], mask_shape[3]

def CreateSegNet(input_shape, n_labels, kernel=3, pool_size=(2, 2), output_mode="softmax"):
	# SegNet implementation from: https://github.com/preddy5/segnet
	# encoder
	inputs = Input(shape=input_shape)

	conv_1 = Convolution2D(64, (kernel, kernel), padding="same")(inputs)
	conv_1 = BatchNormalization()(conv_1)
	conv_1 = Activation("relu")(conv_1)
	conv_2 = Convolution2D(64, (kernel, kernel), padding="same")(conv_1)
	conv_2 = BatchNormalization()(conv_2)
	conv_2 = Activation("relu")(conv_2)

	pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)

	conv_3 = Convolution2D(128, (kernel, kernel), padding="same")(pool_1)
	conv_3 = BatchNormalization()(conv_3)
	conv_3 = Activation("relu")(conv_3)
	conv_4 = Convolution2D(128, (kernel, kernel), padding="same")(conv_3)
	conv_4 = BatchNormalization()(conv_4)
	conv_4 = Activation("relu")(conv_4)

	pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)

	conv_5 = Convolution2D(256, (kernel, kernel), padding="same")(pool_2)
	conv_5 = BatchNormalization()(conv_5)
	conv_5 = Activation("relu")(conv_5)
	conv_6 = Convolution2D(256, (kernel, kernel), padding="same")(conv_5)
	conv_6 = BatchNormalization()(conv_6)
	conv_6 = Activation("relu")(conv_6)
	conv_7 = Convolution2D(256, (kernel, kernel), padding="same")(conv_6)
	conv_7 = BatchNormalization()(conv_7)
	conv_7 = Activation("relu")(conv_7)

	pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_7)

	conv_8 = Convolution2D(512, (kernel, kernel), padding="same")(pool_3)
	conv_8 = BatchNormalization()(conv_8)
	conv_8 = Activation("relu")(conv_8)
	conv_9 = Convolution2D(512, (kernel, kernel), padding="same")(conv_8)
	conv_9 = BatchNormalization()(conv_9)
	conv_9 = Activation("relu")(conv_9)
	conv_10 = Convolution2D(512, (kernel, kernel), padding="same")(conv_9)
	conv_10 = BatchNormalization()(conv_10)
	conv_10 = Activation("relu")(conv_10)

	pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_10)

	conv_11 = Convolution2D(512, (kernel, kernel), padding="same")(pool_4)
	conv_11 = BatchNormalization()(conv_11)
	conv_11 = Activation("relu")(conv_11)
	conv_12 = Convolution2D(512, (kernel, kernel), padding="same")(conv_11)
	conv_12 = BatchNormalization()(conv_12)
	conv_12 = Activation("relu")(conv_12)
	conv_13 = Convolution2D(512, (kernel, kernel), padding="same")(conv_12)
	conv_13 = BatchNormalization()(conv_13)
	conv_13 = Activation("relu")(conv_13)

	pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_13)
	print("Build enceder done..")

	# decoder

	unpool_1 = MaxUnpooling2D(pool_size)([pool_5, mask_5])

	conv_14 = Convolution2D(512, (kernel, kernel), padding="same")(unpool_1)
	conv_14 = BatchNormalization()(conv_14)
	conv_14 = Activation("relu")(conv_14)
	conv_15 = Convolution2D(512, (kernel, kernel), padding="same")(conv_14)
	conv_15 = BatchNormalization()(conv_15)
	conv_15 = Activation("relu")(conv_15)
	conv_16 = Convolution2D(512, (kernel, kernel), padding="same")(conv_15)
	conv_16 = BatchNormalization()(conv_16)
	conv_16 = Activation("relu")(conv_16)

	unpool_2 = MaxUnpooling2D(pool_size)([conv_16, mask_4])

	conv_17 = Convolution2D(512, (kernel, kernel), padding="same")(unpool_2)
	conv_17 = BatchNormalization()(conv_17)
	conv_17 = Activation("relu")(conv_17)
	conv_18 = Convolution2D(512, (kernel, kernel), padding="same")(conv_17)
	conv_18 = BatchNormalization()(conv_18)
	conv_18 = Activation("relu")(conv_18)
	conv_19 = Convolution2D(256, (kernel, kernel), padding="same")(conv_18)
	conv_19 = BatchNormalization()(conv_19)
	conv_19 = Activation("relu")(conv_19)

	unpool_3 = MaxUnpooling2D(pool_size)([conv_19, mask_3])

	conv_20 = Convolution2D(256, (kernel, kernel), padding="same")(unpool_3)
	conv_20 = BatchNormalization()(conv_20)
	conv_20 = Activation("relu")(conv_20)
	conv_21 = Convolution2D(256, (kernel, kernel), padding="same")(conv_20)
	conv_21 = BatchNormalization()(conv_21)
	conv_21 = Activation("relu")(conv_21)
	conv_22 = Convolution2D(128, (kernel, kernel), padding="same")(conv_21)
	conv_22 = BatchNormalization()(conv_22)
	conv_22 = Activation("relu")(conv_22)

	unpool_4 = MaxUnpooling2D(pool_size)([conv_22, mask_2])

	conv_23 = Convolution2D(128, (kernel, kernel), padding="same")(unpool_4)
	conv_23 = BatchNormalization()(conv_23)
	conv_23 = Activation("relu")(conv_23)
	conv_24 = Convolution2D(64, (kernel, kernel), padding="same")(conv_23)
	conv_24 = BatchNormalization()(conv_24)
	conv_24 = Activation("relu")(conv_24)

	unpool_5 = MaxUnpooling2D(pool_size)([conv_24, mask_1])

	conv_25 = Convolution2D(64, (kernel, kernel), padding="same")(unpool_5)
	conv_25 = BatchNormalization()(conv_25)
	conv_25 = Activation("relu")(conv_25)

	conv_26 = Convolution2D(n_labels, (1, 1), padding="valid")(conv_25)
	conv_26 = BatchNormalization()(conv_26)
	conv_26 = Reshape((input_shape[0] * input_shape[1], n_labels), input_shape=(input_shape[0], input_shape[1], n_labels))(conv_26)

	outputs = Activation(output_mode)(conv_26)
	print("Build decoder done..")

	segnet = Model(inputs=inputs, outputs=outputs, name="SegNet")

	return segnet

if __name__=='__main__':
	print("STARTING...")
	model_arch = sys.argv[2]
	if(model_arch=="segnet"):
		input_shape=(256, 256, 3)	# SegNet
	elif(model_arch=="pspnet"):
		input_shape=(473, 473, 3)	# PSPNet
	nb_classes = 21				# PASCAL VOC 2012
	epochs = 100
	batch_size = 5
	learning_rate = 0.05
	width = input_shape[0]
	height = input_shape[1]

    voc_loc = "/media/francis/datasets/_AADatasets/VOC2012_orig/"

	# GET TEST AND VALIDATION DATA
	print("[1] FETCHING DATA...")
	ilan_kukunin = 2
	folder = sys.argv[1]
	Xtest = getData('X'+folder, ilan_kukunin)
	Ytest = getData('Y'+folder, ilan_kukunin)
	print(Xtest.shape)
	print(Ytest.shape)

	# load json and create model
	print("[2] LOADING MODEL...")
	if(model_arch=="segnet"):
		loaded_model = CreateSegNet(input_shape, nb_classes)
	elif(model_arch=="pspnet"):
		loaded_model = pspnet_santelices(input_shape, nb_classes)
	loaded_model.summary()
	# load weights into new model
	loaded_model.load_weights("model_pspnet_santelices_1228_segnet.h5")
	print("Loaded model from disk")
	# Re-compile
	loaded_model.compile(optimizer=SGD(lr=learning_rate, momentum=0.9, nesterov=True),
		loss='categorical_crossentropy',
		metrics=['accuracy'])

	print("[3] EVALUATION OF MODEL ON TEST DATA...")
	score = loaded_model.evaluate(Xtest, Ytest, batch_size=batch_size, verbose = 1)
	print('%s: %0.2f' % (loaded_model.metrics_names[0], score[0]))
	print('%s: %0.2f' % (loaded_model.metrics_names[1], score[1]))

	# Testing
	from sklearn.metrics import classification_report
	from matplotlib import cm

	# Pick random sample
	rand = np.random.randint(len(Xtest))
	print('----------------------')
	print(rand)
	print('----------------------')
	
	# Load input
	filename = folder+'.txt'
	with open (voc_loc+filename, "r") as myfile:
		in_list = myfile.readlines()
		in_list = [item.strip() for item in in_list]
	locx = voc_loc+"JPEGImages/"+in_list[rand]+".jpg"
	x = cv2.resize(plt.imread(locx), (width, height))
	
	# Load ground truth labels
	locy = voc_loc+"SegmentationClass_back/"+in_list[rand]+".png_e"
	vis_y = cv2.resize(plt.imread(locy), (width, height))
	
	# Apply colormap to prediction
	output = loaded_model.predict(np.expand_dims(x, axis=0), verbose=0)[0]
	output = np.reshape(output,(width,height,21))
	vis_pred = apply_colormap(np.argmax(output,axis=-1))
	
	# Display probabilities
	for cnt in range(21):
		maxval = np.max(output[:,:,cnt])*100
		if(maxval>=50):
			print("%s: %2f" % (label_names[cnt],maxval))

	# Plot prediction results
	f, axarr = plt.subplots(3, sharey=True)
	axarr[0].imshow(x)
	axarr[1].imshow(vis_y)
	axarr[2].imshow(vis_pred)

	print("EVALUATION DONE")
	plt.show()
	print("Exiting...")