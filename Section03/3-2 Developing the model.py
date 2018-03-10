#
# Packt Publishing
# Hands-on Tensorflow Lite for Intelligent Mobile Apps
# @author: Juan Miguel Valverde Martinez
#
# Section 3: Handwriting recognition
# Video 3-2: Developing the model
#

import tensorflow as tf
import numpy as np
from scipy.io import loadmat
import itertools

# Import E-MNIST data
mydata = loadmat("matlab/emnist-balanced.mat")

b_size = 10
img_height = 28
img_width = 28
classes = 47
epochs = 1

def hotvector(vector):
	''' This function will transform a vector of labels into a vector of
		one-hot vectors.
	'''

	result = np.zeros((vector.shape[0],classes))
	for i in range(vector.shape[0]):
		result[i][vector[i]]=1
	return result


def thresholdvector(vector):
	''' This function will threshold a vector: 0 and non-zero
		Non-zero values will be transformed into 1's
	'''
	for i in range(vector.shape[0]):
		vector[i]=1.0*(vector[i]>0)
	return vector


def getModel(config):
	init,lr,_ = config

	# Input: 28x28
	xi = tf.placeholder(tf.float32,[None, img_height*img_width],name="inputX")
	yi = tf.placeholder(tf.float32,[None, classes],name="outputY")

	x = tf.reshape(xi, [-1, img_height, img_width, 1])

	with tf.variable_scope("conv1") as scope:
		# First 2D convolution, 
		W = tf.get_variable("W",shape=[3,3,1,32],initializer=init)
		b = tf.get_variable("b",initializer=tf.zeros([32]))

		conv = tf.nn.conv2d(x,W,strides=[1, 1, 1, 1],padding="VALID")
		pre_act = tf.nn.bias_add(conv,b)
		act = tf.nn.relu(pre_act)


	with tf.variable_scope("conv2") as scope:
		W = tf.get_variable("W",shape=[3,3,32,64],initializer=init)
		b = tf.get_variable("b",initializer=tf.zeros([64]))

		conv = tf.nn.conv2d(act,W,strides=[1, 1, 1, 1],padding="VALID")
		pre_act = tf.nn.bias_add(conv,b)
		act = tf.nn.relu(pre_act)

	# Maxpooling
	l3_mp = tf.nn.max_pool(act,[1,2,2,1],strides=[1,2,2,1],padding="VALID")

	# Dense
	l4 = tf.reshape(l3_mp,[-1, 12*12*64])

	with tf.variable_scope("dense1") as scope:
		W = tf.get_variable("W",shape=[12*12*64,128],initializer=init)
		b = tf.get_variable("b",initializer=tf.zeros([128]))

		dense = tf.matmul(l4,W)+b
		act = tf.nn.relu(dense)

	with tf.variable_scope("dense2") as scope:
		W = tf.get_variable("W",shape=[128,classes],initializer=init)
		b = tf.get_variable("b",initializer=tf.zeros([classes]))

		dense = tf.matmul(act,W)+b

	# Prediction. We actually don't ned it
	eval_pred = tf.nn.softmax(dense,name="prediction")

	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=dense,labels=yi)
	cost = tf.reduce_mean(cross_entropy)
	train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
	saver = tf.train.Saver()

	return (xi,yi),train_step,cost,eval_pred,saver


