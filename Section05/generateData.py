#
# Packt Publishing
# Hands-on Tensorflow Lite for Intelligent Mobile Apps
# @author: Juan Miguel Valverde Martinez
#
# Section 5: Gesture recognition
# Video 5-3: Parameter study and data augmentation
#

from PIL import Image
import numpy as np
import scipy.misc
import os

def hotvector(vector,classes):
	''' This function will transform a vector of labels into a vector of
		one-hot vectors.
	'''
	import numpy as np
	result = np.zeros((vector.shape[0],classes))
	for i in range(vector.shape[0]):
		result[i][vector[i]]=1
	return result

def splitDataset(data,splits):
	import numpy as np
	import random
	''' This function will generate Training, Testing and Validation sets.
		Data must be an np object array where the first element is the data
		and the second element are the labels. I.e:

			>> data = np.empty(2,dtype=object)
			>> data[0] = actualInputData
			>> data[1] = labels

		Other example:
			>> totalData = getDataset("emnist")
			>> data = divideDatasets(totalData,[0.8,0.2,0])

	    Splits is a list of percentages, i.e. [0.7,0.2,0.1], corresponding to
		70% training, 20% testing and 10% validation data respectively.
		At first, it collects the data equally spaced from "data".
		Finally, it	shuffles it.
	'''
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


def img2np(image,channels=1):
	im_arr = np.fromstring(image.tobytes(),dtype=np.uint8)
	if channels==1:
		im_arr = im_arr.reshape((image.size[1],image.size[0]))
	else:
		im_arr = im_arr.reshape((image.size[1],image.size[0],channels))
	im_arr = np.array(im_arr,dtype=int)
	return im_arr


def autocrop(matrix):

	# Detect top, bottom, left and right borders
	for i in range(matrix.shape[0]):
		if np.sum(matrix[i,:]==1)>0:
			topVal = i
			break
	for i in range(matrix.shape[0]-1,0,-1):
		if np.sum(matrix[i,:]==1)>0:
			bottomVal = i
			break
	for j in range(matrix.shape[1]):
		if np.sum(matrix[:,j]==1)>0:
			leftVal = j
			break
	for j in range(matrix.shape[1]-1,0,-1):
		if np.sum(matrix[:,j]==1)>0:
			rightVal = j
			break

	# Crop hand
	matrix = matrix[topVal:bottomVal,leftVal:rightVal]
	maxDist = matrix.shape[0] if matrix.shape[0]>matrix.shape[1] else matrix.shape[1]
	new_matrix = np.zeros((maxDist,maxDist))
	offsetX = (maxDist-matrix.shape[1])/2
	offsetY = (maxDist-matrix.shape[0])/2

	# Make a square matrix
	new_matrix[offsetY:offsetY+matrix.shape[0],offsetX:offsetX+matrix.shape[1]]=matrix

	return new_matrix

def getDataset(TARGET_DIM,folder):
	data = []
	labels = []
	im_back = img2np(Image.open(folder+"back.jpg").convert("RGBA"),4)[:,:,:-1]
	mapLabels = [0,0,0,1,1,1,2,2,2]
	for imi in range(1,len(os.listdir(folder))):
		print("Image: "+str(imi))
		im_fore = img2np(Image.open(folder+str(imi)+".jpg").convert("RGBA"),4)[:,:,:-1]
		res = np.zeros((im_back.shape[0],im_back.shape[1]))
		for i in range(res.shape[0]):
			for j in range(res.shape[1]):
				if np.sum(np.abs(im_back[i,j,:]-im_fore[i,j,:]))>50:
					res[i,j]=1

		# Autocrop (get only the hand)
		matrix = autocrop(res)
		# Convert to Image
		im_orig = Image.fromarray(matrix).convert("L")
	
		for deg in range(-45,50,5):
			# Rotate
			im = im_orig.rotate(deg,expand=1)
			# Convert to matrix
			res = img2np(im)
			# Autocrop again (rotating will add more pixels)
			matrix = autocrop(res)
			# Resize to fit the model
			im = im.resize((TARGET_DIM,TARGET_DIM))

			im = img2np(im)

			data.append(im)
			labels.append(mapLabels[imi-1])

	total = np.empty(2,dtype=object)
	total[0] = np.array(data)
	total[1] = np.array(labels)
	return total


