#
# Pack#t Publishing
# Hands-on Tensorflow Lite for Intelligent Mobile Apps.
# @author: Juan Miguel Valverde Martinez
#
# Section 6: Voice recognition
# Video 6-3: Dropout and dataset generation
#

from scipy.io.wavfile import read
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import os
from collections import deque

STRIDES = 100
num = 10000

def hotvector(vector,classes):
	''' This function will transform a vector of labels into a vector of
		one-hot vectors.
	'''
	import numpy as np
	result = np.zeros((vector.shape[0],classes))
	for i in range(vector.shape[0]):
		result[i][vector[i]]=1
	return result

def divideDatasets(data,splits):
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
        Finally, it shuffles it.
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


def normalizeAudio(num,audio):
	cmax = np.max(audio)
	cmin = np.min(audio)

	if len(audio)<num:
		diff = num-len(audio)
		final = np.zeros(num)
		final[diff/2:-diff/2]=audio
		audio = final
	else:
		indicesVector = np.array([i*len(audio)//num + len(audio)//(2*num) for i in range(num)])
		audio = audio[indicesVector]

	audio = (cmax-audio)/(cmax-cmin)
	return audio

def getDataset(num,folder):
	data = []
	labels = []
	mapLabels = [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4]

	#for soi in range(1,len(os.listdir(folder))+1):
	for soi in range(1,26):
		#vector = read(folder+"filename"+str(soi)+".wav")[1]
		vector = read(folder+str(soi)+".wav")[1]*1.0

		vector = normalizeAudio(num,vector)

		items = deque(vector)
		for i in range(num/STRIDES):
			items.rotate(STRIDES)
			data.append(np.array(items))
			labels.append(mapLabels[soi-1])

	total = np.empty(2,dtype=object)
	total[0] = np.array(data)
	total[1] = np.array(labels)
	return total
		

'''
#a = np.reshape(vector,(100,100))
a=vector
plt.plot(a)
plt.show()
'''
