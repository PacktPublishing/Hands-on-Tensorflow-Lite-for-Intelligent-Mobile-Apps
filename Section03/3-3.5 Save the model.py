# 
# Packt Publishing
# Hands-on Tensorflow Lite for Intelligent Mobile Apps
# @author: Juan Miguel Valverde Martinez
#
# Section 3: Developing our first model in Tensorflow Lite
# Video 3-3.5: Save the model using the best configuration
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


conf = [tf.keras.initializers.he_uniform(),0.000505,2]
(xi,yi),optimizer,cost,eval_pred,saver = getModel(conf)
b_size = conf[-1]

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	# Training
	for i in range(epochs):
		# Batches
		for j in range(0,mydata["dataset"]["train"][0][0][0][0][1].shape[0],b_size):
			x_raw = thresholdvector(mydata["dataset"]["train"][0][0][0][0][0][j:j+b_size])
			y_raw = hotvector(mydata["dataset"]["train"][0][0][0][0][1][j:j+b_size])


			[la,c]=sess.run([optimizer,cost], feed_dict={xi: x_raw, yi: y_raw})

	## Saving the graph for later
	'''
	saver.save(sess, 'tmp/my-weights')
	g = sess.graph
	gdef = g.as_graph_def()
	tf.train.write_graph(gdef,"tmp","graph.pb",False)
	'''
	# Testing
	c=0;g=0
	for i in range(mydata["dataset"]["test"][0][0][0][0][1].shape[0]):
		x_raw = thresholdvector(mydata["dataset"]["test"][0][0][0][0][0][i:i+1]) # It will just have the proper shape
		y_raw = hotvector(mydata["dataset"]["test"][0][0][0][0][1][i:i+1])

		pred=sess.run(eval_pred,feed_dict={xi: x_raw})

		if np.argmax(y_raw)==np.argmax(pred):
			g+=1
			
		c+=1

	acc=1.0*g/c

	print("Accuracy: "+str(acc))


