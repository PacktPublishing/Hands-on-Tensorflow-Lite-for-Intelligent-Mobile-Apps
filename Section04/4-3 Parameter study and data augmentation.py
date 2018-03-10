#
# Packt Publishing
# Hands-on Tensorflow Lite for Intelligent Mobile Apps
# @author: Juan Miguel Valverde Martinez
#
# Section 4: Pattern recognition
# Video 4-3: Parameter study and data augmentation
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


initP = [tf.contrib.layers.xavier_initializer(),tf.keras.initializers.he_uniform()]
lrP = np.linspace(0.000001,0.0001,10)
batchP = [2,4,8]
parameters = [initP,lrP,batchP] # 60 combinations
allConfgs = list(itertools.product(*parameters))

data = getDataset(TARGET_DIM,"imgs/")
data = divideDatasets(data,[0.7,0.2,0.1])

counter=0
for conf in allConfgs:
	counter+=1
	print(counter,len(allConfgs))
	(x,y),optimizer,cost,eval_pred,saver = getModel(conf)
	b_size  = conf[-1]
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		merged = tf.summary.merge_all()
		writer = tf.summary.FileWriter("logs/"+conf2str(conf),sess.graph)
		step = 0
		# Training
		for i in range(epochs):
			# Batches
			for j in range(0,data["train"][1].shape[0],b_size):
				x_raw = data["train"][0][j:j+b_size]
				y_raw = hotvector(data["train"][1][j:j+b_size],classes)

				[la,c,summary]=sess.run([optimizer,cost,merged], feed_dict={x: x_raw, y: y_raw})
				writer.add_summary(summary,step)
				step+=1

		writer.close()

		#saver.save(sess, 'tmp/my-weights')
		#g = sess.graph
		#gdef = g.as_graph_def()
		#tf.train.write_graph(gdef,"tmp","graph.pb",False)

		# Testing
		c=0;g=0
		goodones=[]
		preds = np.zeros(classes)
		for i in range(data["test"][1].shape[0]):
			x_raw = data["test"][0][i:i+1]
			y_raw = hotvector(data["test"][1][i:i+1],classes)

			pred=sess.run(eval_pred,feed_dict={x: x_raw})
			preds[np.argmax(pred)]+=1

			if np.argmax(y_raw)==np.argmax(pred):
				g+=1
				if np.argmax(pred)==3:
					goodones.append(x_raw)
				
			c+=1

		acc = 1.0*g/c
		print("Accuracy: "+str(acc))

	f=open("logRes","a")
	f.write("{0},{1},{2},{3}\n".format(str(conf[0]),conf[1],conf[2],str(acc)))
	f.close()

	tf.reset_default_graph()
