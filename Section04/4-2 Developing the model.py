#
# Packt Publishing
# Hands-on Tensorflow Lite for Intelligent Mobile Apps
# @author: Juan Miguel Valverde Martinez
#
# Section 4: Pattern recognition
# Video 4-2: Developing the model
#

import tensorflow as tf
import numpy as np
from generateData import getDataset,divideDatasets,hotvector
from PIL import Image
import itertools

TARGET_DIM = 40

epochs = 4
classes = 13
def conf2str(conf):
	res = ""
	if str(conf[0]).find('_initializer')!=-1:
		res+="_initxavier"
	else:
		res+="_inithe"

	res+="_lr"+str(conf[1])
	res+="_bs"+str(conf[2])
	return res

def getModel(config):
	init,lr,_ = config

	# Input: 40x40x3
	x = tf.placeholder(tf.float32,[None, TARGET_DIM, TARGET_DIM, 3],name="inputX")
	y = tf.placeholder(tf.float32,[None, classes],name="outputY")

	with tf.variable_scope("conv1") as scope:
		# First 2D convolution, 
		W = tf.get_variable("W",shape=[3,3,3,16],initializer=init)
		b = tf.get_variable("b",initializer=tf.zeros([16]))

		conv = tf.nn.conv2d(x,W,strides=[1, 1, 1, 1],padding="VALID")
		pre_act = tf.nn.bias_add(conv,b)
		act_ = tf.nn.relu(pre_act)

	with tf.variable_scope("branch1") as sc:
		with tf.variable_scope("conv2") as scope:
			# First 2D convolution, 
			W = tf.get_variable("W",shape=[3,3,16,32],initializer=init)
			b = tf.get_variable("b",initializer=tf.zeros([32]))
			conv = tf.nn.conv2d(act_,W,strides=[1, 1, 1, 1],padding="VALID")
			pre_act = tf.nn.bias_add(conv,b)
			act = tf.nn.relu(pre_act)

		l3_mp1 = tf.nn.max_pool(act,[1,2,2,1],strides=[1,2,2,1],padding="VALID")

	with tf.variable_scope("branch2") as sc:
		with tf.variable_scope("conv2") as scope:
			# First 2D convolution, 
			W = tf.get_variable("W",shape=[3,3,16,32],initializer=init)
			b = tf.get_variable("b",initializer=tf.zeros([32]))
			conv = tf.nn.conv2d(act_,W,strides=[1, 1, 1, 1],padding="VALID")
			pre_act = tf.nn.bias_add(conv,b)
			act = tf.nn.relu(pre_act)

		l3_mp2 = tf.nn.max_pool(act,[1,2,2,1],strides=[1,2,2,1],padding="VALID")

	end = tf.concat([l3_mp1,l3_mp2],1)

	l5 = tf.reshape(end,[-1, 2*18*18*32])

	with tf.variable_scope("dense") as scope:
		W = tf.get_variable("W",shape=[2*18*18*32,classes],initializer=init)
		b = tf.get_variable("b",initializer=tf.zeros([classes]))

		dense = tf.matmul(l5,W)+b

	eval_pred = tf.nn.softmax(dense,name="prediction")

	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=dense,labels=y)
	cost = tf.reduce_mean(cross_entropy)
	train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
	saver = tf.train.Saver()
	tf.summary.scalar("cost",cost)

	return (x,y),train_step,cost,eval_pred,saver


