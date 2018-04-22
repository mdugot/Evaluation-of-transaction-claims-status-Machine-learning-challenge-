import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

class NeuralNetwork:

	def __init__(self, data, hiddenLayers, key = None):

		self.data = data
		self.key = key
		outputlen = self.data.outputlen
		if (key):
			outputlen = 1

		self.inputs = tf.placeholder(dtype = tf.float32, shape=[None, self.data.inputlen])
		self.labels = tf.placeholder(dtype = tf.float32, shape=[None, outputlen])
		self.learningRate = tf.placeholder(dtype = tf.float32, shape=[])
		self.dropProb = tf.placeholder(dtype = tf.float32, shape=[])
		self.training = tf.placeholder(dtype = tf.bool, shape=[])

		self.layers = []
		self.hiddenLayers = hiddenLayers
		lastLayer = self.inputs
		for layer in hiddenLayers:
			print("add hidden layer")
			self.layers.append(tf.layers.dense(lastLayer, layer, tf.nn.relu))
			self.layers.append(tf.layers.dropout(self.layers[-1], self.dropProb, training=self.training))
			lastLayer = self.layers[-1]

		if key:
			print("ONE BY ONE")
			self.outputs = tf.layers.dense(lastLayer, outputlen, tf.nn.relu)
			self.cost = tf.losses.hinge_loss(self.labels, self.outputs)
			self.prediction = self.outputs
		else:
			print("MULTI LABELS")
			self.outputs = tf.layers.dense(lastLayer, outputlen)
			self.softmax = tf.nn.softmax(self.outputs)
			self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.labels, logits = self.outputs))
			self.prediction = tf.argmax(self.outputs, 1)

		self.optim = tf.train.AdamOptimizer(self.learningRate).minimize(self.cost)

	def start(self):
		self.session = tf.Session()
		self.session.run(tf.global_variables_initializer())
		return self.session

	def train(self, batch, iterations, learningRate, dropProb):

		for i in tqdm(range(iterations)):
			batchData = self.data.randomBatch(batch)
			#batchData = self.data.balancedBatch(batch)

			self.session.run(self.optim, feed_dict={
				self.inputs: batchData["inputs"],
				self.labels: batchData["outputs"],
				self.learningRate: learningRate,
				self.dropProb: dropProb,
				self.training: True
			})
		print("TRAINING")
		self.check(self.data.training, self.data.trainingLabels)
		print("VALIDATION")
		self.check(self.data.validation, self.data.validationLabels)

	def trainOneByOne(self, batch, iterations, learningRate, dropProb):

		for i in tqdm(range(iterations)):
			batchData = self.data.versusBatch(batch, self.key)

			self.session.run(self.optim, feed_dict={
				self.inputs: batchData["inputs"],
				self.labels: batchData["outputs"],
				self.learningRate: learningRate,
				self.dropProb: dropProb,
				self.training: True
			})
		print("TRAINING")
		self.checkOneByOne(5000)

	def check(self, inputs, labels):

		predictions,cost = self.session.run([self.prediction, self.cost], feed_dict={
			self.inputs: inputs,
			self.labels: labels,
			self.dropProb: 0,
			self.training: False
		})
		success = 0
		for i in range(len(predictions)):
			if labels[i][predictions[i]] == 1:
				success = success + 1
		print("cost : " + str(cost))
		print("success : " + str(success) + "/" + str(len(predictions)))
		print("rate : " + str(success / len(predictions) * 100) + "%")

	def checkOneByOne(self, size):

		batchData = self.data.versusBatch(size, self.key)
		predictions,cost = self.session.run([self.prediction, self.cost], feed_dict={
			self.inputs: batchData["inputs"],
			self.labels: batchData["outputs"],
			self.dropProb: 0,
			self.training: False
		})
		success = 0
		predictions[predictions >= 0.5] = 1
		predictions[predictions < 0.5] = 0
		for i in range(len(predictions)):
			if batchData["outputs"][i][0] == predictions[i]:
				success = success + 1
		print("cost : " + str(cost))
		print("success : " + str(success) + "/" + str(len(predictions)))
		print("rate : " + str(success / len(predictions) * 100) + "%")

	def sampleResult(self, batch):
		if self.key:
			batchData = self.data.versusBatch(batch, self.key)
		else:
			batchData = self.data.balancedBatch(batch)
		return self.session.run([self.prediction, self.outputs, self.cost], feed_dict={
			self.inputs: batchData["inputs"],
			self.labels: batchData["outputs"],
			self.dropProb: 0,
			self.training: False
		})

	def save(self, filename):

		result = np.zeros([len(self.data.testing)], dtype="int32,U30")
		for i in tqdm(range(len(self.data.testing))):
			predictions = self.session.run(self.prediction, feed_dict={
					self.inputs: [self.data.testing[i]],
					self.dropProb: 0,
					self.training: False
			})
			result[i][0] = self.data.dataframe_test["ID"][i]
			result[i][1] = self.data.getClaim(predictions[0])

		np.savetxt(filename, result, "%s", delimiter=";", header = "ID;CLAIM_TYPE", comments="")




