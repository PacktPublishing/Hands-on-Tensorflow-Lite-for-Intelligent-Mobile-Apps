#
# Packt Publishing
# Hands-on Tensorflow Lite for Intelligent Mobile Apps
# @author: Juan Miguel Valverde Martinez
#
# Section 3: Handwriting recognition
# Video 3-4: Testing the model
#


import tensorflow as tf
import numpy as np
from PIL import Image

def read_image(path):
	''' This function will read an image, threshold it and convert it to the
		right format for our network.
	'''
	# Read the file
	image = Image.open(path).convert("L")
	im = np.fromstring(image.tobytes(), dtype=np.uint8)
	# First we convert it into a numpy array, same size as the image
	im = im.reshape((image.size[1],image.size[0]))
	res_im = np.zeros_like(im)
	# Change axis of pixels (same format as the training images!)
	for i in range(im.shape[0]):
		for j in range(im.shape[1]):
			res_im[j,i] = im[i,j]
	# Reshape to the appropriate format for the network
	res_im = res_im.reshape((1,image.size[1]*image.size[0]))
	# Threshold, remember: 0 background, 1 character
	res_im = 1.0*(res_im>127)
	return res_im

# Read the graph
with tf.gfile.GFile("tmp/graph.pb", "rb") as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())

with tf.Session() as sess:
	# Load weights
	saver = tf.train.import_meta_graph("tmp/my-weights.meta")
	saver.restore(sess,tf.train.latest_checkpoint("./tmp"))
	graph = tf.get_default_graph()

	# Check the operations in graph.get_operations() and add :0 (tensor)
	x = graph.get_tensor_by_name("inputX:0")
	eval_pred = graph.get_tensor_by_name("prediction:0")

	labels = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","a","b","d","e","f","g","h","n","q","r","t"]

	# Testing
	x_raw = read_image("letter.png")

	# Prediction
	pred=sess.run(eval_pred,feed_dict={x: x_raw})

	# Reversed indices sorted by probability (highest -> first)
	indices = np.argsort(pred)[0][::-1]

	# Print the results sorted
	result = ""
	for i in range(len(labels)):
		result+=labels[indices[i]]+","
	print("Result")
	print(result)
	



	


