#
# Packt Publishing
# Hands-on Tensorflow Lite for Intelligent Mobile Apps
# @author: Juan Miguel Valverde Martinez
#
# Section 5: Gesture recognition
# Video 5-3: Parameter study and data augmentation
#

import tensorflow as tf
import numpy as np
import random
import scipy.misc
from generateData import getDataset,splitDataset,hotvector
from PIL import Image
import itertools

def crossValidation(rows,folds):
	indices = [i for i in range(rows)]
	random.shuffle(indices)
	sizeFold = int(rows/folds)

	res = []
	for i in range(folds):
		testIndices = indices[i:i+sizeFold]
		trainIndices = list(set(indices)-set(testIndices))
		res.append([trainIndices,testIndices])

	return res

TARGET_DIM = 60

epochs = 4
classes = 3

def getModel(config):
	init,lr,_ = config

	# Input: 60x60
	x = tf.placeholder(tf.float32,[None, TARGET_DIM, TARGET_DIM],name="inputX")
	y = tf.placeholder(tf.float32,[None, classes],name="outputY")
	phase = tf.placeholder(tf.bool, name="phase")

	sum1 = tf.reduce_sum(x,axis=1)
	sum2 = tf.reduce_sum(x,axis=2)
	total_sum = tf.concat([sum1,sum2],axis=1) #-1,TARGET_DIM*2

	with tf.variable_scope("dense_branch") as scope:
		W = tf.get_variable("W",shape=[TARGET_DIM*2,200],initializer=init)
		b = tf.get_variable("b",initializer=tf.zeros([200]))
	
		layer = tf.add(tf.matmul(total_sum,W),b)
		batch_n = tf.contrib.layers.batch_norm(layer,is_training=phase)
		layer = tf.nn.sigmoid(batch_n)

	xi = tf.reshape(x,[-1,TARGET_DIM,TARGET_DIM,1])

	with tf.variable_scope("conv1") as scope:
		# First 2D convolution, 
		W = tf.get_variable("W",shape=[3,3,1,16],initializer=init)
		b = tf.get_variable("b",initializer=tf.zeros([16]))

		conv = tf.nn.conv2d(xi,W,strides=[1, 1, 1, 1],padding="VALID")
		pre_act = tf.nn.bias_add(conv,b)
		batch_n = tf.contrib.layers.batch_norm(pre_act,is_training=phase)
		act_ = tf.nn.relu(batch_n)

	with tf.variable_scope("branch1") as sc:
		with tf.variable_scope("conv2") as scope:
			# First 2D convolution, 
			W = tf.get_variable("W",shape=[3,3,16,32],initializer=init)
			b = tf.get_variable("b",initializer=tf.zeros([32]))
			conv = tf.nn.conv2d(act_,W,strides=[1, 1, 1, 1],padding="VALID")
			pre_act = tf.nn.bias_add(conv,b)
			batch_n = tf.contrib.layers.batch_norm(pre_act,is_training=phase)
			act = tf.nn.relu(batch_n)

		l3_mp1 = tf.nn.max_pool(act,[1,2,2,1],strides=[1,2,2,1],padding="VALID")

	with tf.variable_scope("branch2") as sc:
		with tf.variable_scope("conv2") as scope:
			# First 2D convolution, 
			W = tf.get_variable("W",shape=[3,3,16,32],initializer=init)
			b = tf.get_variable("b",initializer=tf.zeros([32]))
			conv = tf.nn.conv2d(act_,W,strides=[1, 1, 1, 1],padding="VALID")
			pre_act = tf.nn.bias_add(conv,b)
			batch_n = tf.contrib.layers.batch_norm(pre_act,is_training=phase)
			act = tf.nn.relu(batch_n)

		l3_mp2 = tf.nn.max_pool(act,[1,2,2,1],strides=[1,2,2,1],padding="VALID")

	end = tf.concat([l3_mp1,l3_mp2],1)
	l7 = tf.reshape(end,[-1, 2*28*28*32])
	l8 = tf.concat([l7,layer],axis=1)

	with tf.variable_scope("dense1") as scope:
		W = tf.get_variable("W",shape=[2*28*28*32+200,classes],initializer=init)
		b = tf.get_variable("b",initializer=tf.zeros([classes]))

		dense = tf.matmul(l8,W)+b
		batch_n = tf.contrib.layers.batch_norm(dense,is_training=phase)

	eval_pred = tf.nn.softmax(batch_n,name="prediction")
	
	with tf.variable_scope("cost"):
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=dense,labels=y)
		cost = tf.reduce_mean(cross_entropy)

	with tf.variable_scope("train"):
		train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

	saver = tf.train.Saver()

	return (x,y,phase),train_step,cost,eval_pred,saver


# 0.00001 --> 0.0001
initP = [tf.contrib.layers.xavier_initializer(),tf.keras.initializers.he_uniform()]
lrP = np.linspace(0.000001,0.0001,10)
batchP = [2,4,8]
parameters = [initP,lrP,batchP]
allConfgs = list(itertools.product(*parameters))

data = getDataset(TARGET_DIM,"imgs/")

folds = 5
counter=0
for conf in allConfgs:
	counter+=1
	print(counter,len(allConfgs))
	(x,y,phase),optimizer,cost,eval_pred,saver = getModel(conf)
	b_size  = conf[-1]

	indices = crossValidation(data[0].shape[0],5)

	accuracies = []
	for k in range(folds):
		print("Fold: {0}".format(k))
		trainingData = data[0][indices[k][0]]
		trainingLabels = data[1][indices[k][0]]
		testingData = data[0][indices[k][1]]
		testingLabels = data[1][indices[k][1]]

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			
			# Training
			for i in range(epochs):
				# Batches
				for j in range(0,trainingData.shape[0],b_size):
					x_raw = trainingData[j:j+b_size]
					y_raw = hotvector(trainingLabels[j:j+b_size],classes)

					[la,c]=sess.run([optimizer,cost], feed_dict={x: x_raw, y: y_raw, phase: True})


			#saver.save(sess, 'tmp/my-weights')
			#g = sess.graph
			#gdef = g.as_graph_def()
			#tf.train.write_graph(gdef,"tmp","graph.pb",False)

			# Testing
			c=0;g=0
			goodones=[]
			preds = np.zeros(classes)
			for i in range(testingData.shape[0]):
				x_raw = testingData[i:i+1]
				y_raw = hotvector(testingLabels[i:i+1],classes)

				pred=sess.run(eval_pred,feed_dict={x: x_raw, phase: False})
				preds[np.argmax(pred)]+=1

				if np.argmax(y_raw)==np.argmax(pred):
					g+=1
					if np.argmax(pred)==3:
						goodones.append(x_raw)
					
				c+=1

			acc = 1.0*g/c
			accuracies.append(acc)

	f=open("logRes_xval","a")
	f.write("{0},{1},{2},{3},{4}\n".format(str(conf[0]),conf[1],conf[2],str(np.mean(accuracies)),str(np.std(accuracies))))
	f.close()

	tf.reset_default_graph()
