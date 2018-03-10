#
# Packt Publishing
# Hands-on Tensorflow Lite for Intelligent Mobile Apps
# @author: Juan Miguel Valverde Martinez
#
# Section 5: Gesture recognition
# Video 5-4.1: Adapting and debugging the model
#

import tensorflow as tf
import numpy as np
import random
import scipy.misc
from generateData import getDataset,splitDataset,hotvector
from PIL import Image
import itertools

TARGET_DIM = 60

epochs = 4
classes = 3

def getModel():
	lr = 8.9e-05
	init = tf.keras.initializers.he_uniform()

	# Input: 60x60
	x = tf.placeholder(tf.float32,[None, TARGET_DIM, TARGET_DIM],name="inputX")
	y = tf.placeholder(tf.float32,[None, classes],name="outputY")
	x_sum1 = tf.placeholder(tf.float32,[None, TARGET_DIM],name="sum1")
	x_sum2 = tf.placeholder(tf.float32,[None, TARGET_DIM],name="sum2")

	total_sum = tf.concat([x_sum1,x_sum2],axis=1) #-1,TARGET_DIM*2

	with tf.variable_scope("dense_branch") as scope:
		W = tf.get_variable("W",shape=[TARGET_DIM*2,200],initializer=init)
		b = tf.get_variable("b",initializer=tf.zeros([200]))
	
		layer = tf.add(tf.matmul(total_sum,W),b)
		layer = tf.nn.sigmoid(layer)

	xi = tf.reshape(x,[-1,TARGET_DIM,TARGET_DIM,1])

	with tf.variable_scope("conv1") as scope:
		# First 2D convolution, 
		W = tf.get_variable("W",shape=[3,3,1,16],initializer=init)
		b = tf.get_variable("b",initializer=tf.zeros([16]))

		conv = tf.nn.conv2d(xi,W,strides=[1, 1, 1, 1],padding="VALID")
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
	l7 = tf.reshape(end,[-1, 2*28*28*32])
	l8 = tf.concat([l7,layer],axis=1)

	with tf.variable_scope("dense1") as scope:
		W = tf.get_variable("W",shape=[2*28*28*32+200,classes],initializer=init)
		b = tf.get_variable("b",initializer=tf.zeros([classes]))

		dense = tf.matmul(l8,W)+b

	eval_pred = tf.nn.softmax(dense,name="prediction")
	
	with tf.variable_scope("cost"):
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=dense,labels=y)
		cost = tf.reduce_mean(cross_entropy)

	with tf.variable_scope("train"):
		train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

	saver = tf.train.Saver()
	return (x,x_sum1,x_sum2,y),train_step,cost,eval_pred,saver

b_size  = 2

data = getDataset(TARGET_DIM,"imgs/")
data = splitDataset(data,[0.8,0.2,0])

counter=0
(x,x_sum1,x_sum2,y),optimizer,cost,eval_pred,saver = getModel()

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	# Training
	for i in range(epochs):
		# Batches
		for j in range(0,data["train"][0].shape[0],b_size):
			x_raw = data["train"][0][j:j+b_size]
			x_raw_sum1 = np.sum(x_raw,axis=1)
			x_raw_sum2 = np.sum(x_raw,axis=2)
			y_raw = hotvector(data["train"][1][j:j+b_size],classes)

			[la,c]=sess.run([optimizer,cost], feed_dict={x: x_raw, y: y_raw, x_sum1: x_raw_sum1, x_sum2:x_raw_sum2})


	#saver.save(sess, 'tmp/my-weights')
	#g = sess.graph
	#gdef = g.as_graph_def()
	#tf.train.write_graph(gdef,"tmp","graph.pb",False)

	# Testing
	c=0;g=0
	goodones=[]
	preds = np.zeros(classes)
	for i in range(data["test"][0].shape[0]):
		x_raw = data["test"][0][i:i+1]
		x_raw_sum1 = np.sum(x_raw,axis=1)
		x_raw_sum2 = np.sum(x_raw,axis=2)
		y_raw = hotvector(data["test"][1][i:i+1],classes)

		pred=sess.run(eval_pred,feed_dict={x: x_raw, x_sum1: x_raw_sum1,x_sum2:x_raw_sum2})
		preds[np.argmax(pred)]+=1

		if np.argmax(y_raw)==np.argmax(pred):
			g+=1
			if np.argmax(pred)==3:
				goodones.append(x_raw)
			
		c+=1

	acc = 1.0*g/c
	print("Accuracy: "+str(acc))

