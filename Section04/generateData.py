#
# Packt Publishing
# Hands-on Tensorflow Lite for Intelligent Mobile Apps
# @author: Juan Miguel Valverde Martinez
#
# Section 4: Pattern recognition
# Video 4-3: Parameter study and data augmentation
#

from PIL import Image
import random
import numpy as np
import os

def hotvector(vector,classes):
	""" This function will transform a vector of labels into a vector of
		one-hot vectors.
	"""
	result = np.zeros((vector.shape[0],classes))
	for i in range(vector.shape[0]):
		result[i][vector[i]]=1
	return result

def divideDatasets(data,splits):
	"""
	Splits the dataset
	"""

	if np.sum(np.array(splits)*10)!=10:
		print(np.sum(splits))
		raise Exception("The splits are percentages and must sum 1 in total. Example: [0.7,0.2,0.1]")

	if len(splits)!=3:
		raise Exception("There must be 3 elements in 'splits' corresponding to training, testing and validation splits. If you don't need validation, write 0, [0.8,0.2,0]")

	rows = data[1].shape[0]
	cut1 = int(splits[0]*rows)
	cut2 = int(splits[1]*rows)
	cut3 = int(splits[2]*rows)

	totalIndices = np.array([x for x in range(rows)])
	indicesTraining = np.array([i*rows//cut1 + rows//(2*cut1) for i in range(cut1)])
	restIndices=np.array(sorted(list(set(totalIndices)-set(indicesTraining))))

	if cut3!=0:
		indicesTesting = np.array([i*len(restIndices)//cut2 + len(restIndices)//(2*cut2) for i in range(cut2)])
		indicesTesting = restIndices[indicesTesting]
		indicesValidating = np.array(sorted(list(set(restIndices)-set(indicesTesting))))

		random.shuffle(indicesValidating)
	else:
		indicesTesting = restIndices

	random.shuffle(indicesTraining)
	random.shuffle(indicesTesting)

	newdata = {}
	newdata["train"] = np.empty(2,dtype=object)
	newdata["test"] = np.empty(2,dtype=object)

	newdata["train"][0] = data[0][indicesTraining]
	newdata["train"][1] = data[1][indicesTraining]
	newdata["test"][0] = data[0][indicesTesting]
	newdata["test"][1] = data[1][indicesTesting]

	if cut3!=0:
		newdata["validation"] = np.empty(2,dtype=object)
		newdata["validation"][0] = data[0][indicesValidating]
		newdata["validation"][1] = data[1][indicesValidating]

	return newdata

def img2np(image):
	"""
	Loads JPEG image into 3D Numpy array of shape 
	(width, height, channels)
	"""
	im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
	im_arr = im_arr.reshape((image.size[1], image.size[0], 4))
	return im_arr

def getPortions(im,outputDim):
	import scipy.misc,random
	i=0;found=False
	totalOpacity=outputDim*outputDim*255
	totalPortions = []
	while i<im.shape[0]-outputDim:
		j=0	
		while j<im.shape[1]-outputDim:
			portion = im[i:i+outputDim,j:j+outputDim,:]
			if np.sum(portion[:,:,-1]==0)==0:
				found=True
				totalPortions.append(portion[:,:,:-1])
				j+=outputDim-1
			j+=1
		if found:
			i+=outputDim-1
			found=False
		i+=1

	return totalPortions

def getRatios(size,ratios):
	res = []
	for r in ratios:
		res.append((int(size[0]*r),int(size[1]*r)))
	return res

def getDataset(outputDim,folder):
	data = []
	labels = []
	counter = 0
	for ind in range(len(os.listdir(folder))):
		pathfile = folder+str(ind)+".jpg"
		image_original = Image.open(pathfile).convert("RGBA")
		# Get all sizes combinations
		resizeRatios = getRatios(image_original.size,[0.3,0.4,0.5,0.6])
		for r in resizeRatios:
			image = image_original.resize(r)
			for i in range(0,360,10):
				counter+=1
				# Rotate the image
				rot = image.rotate(i,expand=1)
				# Convert to numpy array
				numpyImage = img2np(rot)
				# Get valid portions of the data
				p = getPortions(numpyImage,outputDim)
				labels.extend([ind for _ in range(len(p))])
				data.extend(p)
	print(np.array(data).shape)
	total = np.empty(2,dtype=object)
	total[0] = np.array(data)
	total[1] = np.array(labels)
	return total

