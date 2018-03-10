#
# Packt Publishing
# Hands-on Tensorflow Lite for Intelligent Mobile Apps
# @author: Juan Miguel Valverde Martinez
#
# Section 4: Pattern recognition
# Video 4-4: Testing the model
#

import tensorflow as tf
import numpy as np
import scipy.misc
from generateData import getDataset,divideDatasets,hotvector
from PIL import Image
import itertools

tf.reset_default_graph()

TARGET_DIM = 40

epochs = 4
classes = 13

def getModel():
	init = tf.contrib.layers.xavier_initializer()
	lr = 8.9e-5

	# Input: 28x28
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

	l7 = tf.reshape(end,[-1, 36*18*32])

	with tf.variable_scope("dense") as scope:
		W = tf.get_variable("W",shape=[36*18*32,classes],initializer=init)
		b = tf.get_variable("b",initializer=tf.zeros([classes]))

		dense = tf.matmul(l7,W)+b

	eval_pred = tf.nn.softmax(dense,name="prediction")

	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=dense,labels=y)
	cost = tf.reduce_mean(cross_entropy)
	train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
	saver = tf.train.Saver()

	return (x,y),train_step,cost,eval_pred,saver



data=getDataset(TARGET_DIM,"imgs/")
data = divideDatasets(data,[0.7,0.2,0.1])
b_size = 2

(x,y),optimizer,cost,eval_pred,saver = getModel()
	
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	# Training
	for i in range(epochs):
		# Batches
		for j in range(0,data["train"][1].shape[0],b_size):
			x_raw = data["train"][0][j:j+b_size]
			y_raw = hotvector(data["train"][1][j:j+b_size],classes)

			[la,c]=sess.run([optimizer,cost], feed_dict={x: x_raw, y: y_raw})
			print("Epoch {0}/{1}. Batch: {2}/{3}. Loss: {4}".format(i+1,epochs,(j+b_size)/b_size,data["train"][1].shape[0]/b_size,c))

	saver.save(sess, 'tmp/my-weights')
	g = sess.graph
	gdef = g.as_graph_def()
	tf.train.write_graph(gdef,"tmp","graph.pb",False)

	# Testing
	c=0;g=0
	goodones=[]
	# Confusion matrix
	pred_sq = np.zeros((classes,classes))
	for i in range(data["test"][1].shape[0]):
		x_raw = data["test"][0][i:i+1]
		y_raw = hotvector(data["test"][1][i:i+1],classes)

		pred=sess.run(eval_pred,feed_dict={x: x_raw})
		pred_sq[np.argmax(y_raw),np.argmax(pred)]+=1

		if np.argmax(y_raw)==np.argmax(pred):
			g+=1
			if np.argmax(pred)==3:
				goodones.append(x_raw)
			
		c+=1

	acc = 1.0*g/c
	print("Accuracy: "+str(acc))

	print(pred_sq)

