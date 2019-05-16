# -*- coding: UTF-8 -*-

import matplotlib.pyplot
import numpy
import scipy.special


class neuralNetwork:
	def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes
		self.lr = learningrate

		#self.wih = (numpy.random.rand(self.hnodes, self.inodes) - 0.5)
		#self.who = (numpy.random.rand(self.onodes, self.hnodes) - 0,5)
		self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5) ,(self.hnodes, self.inodes))
		self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
		#simog激励函数
		self.actication_function = lambda x : scipy.special.expit(x)

		pass


	def train(self, input_list, targets_list):
		# 将输入的list转为2维度数组
		inputs = numpy.array(input_list, ndmin=2).T 
		targets = numpy.array(targets_list, ndmin=2).T

		hidden_input = numpy.dot(self.wih, inputs)
		hidden_output = self.actication_function(hidden_input)

		final_input = numpy.dot(self.who, hidden_output)
		final_output = self.actication_function(final_input)

		output_error = targets - final_output
		hidden_error = numpy.dot(self.who.T, output_error)

		temp = output_error * final_output * (1.0 - final_output)
		self.who += self.lr*numpy.dot( temp, numpy.transpose(hidden_output))

		temp = hidden_error * hidden_output * (1.0 - hidden_output)
		self.wih += self.lr*numpy.dot(temp, numpy.transpose(inputs))

		pass

	# 命名为推理更合适
	def query(self, input_list):
		inputs = numpy.array(input_list, ndmin=2).T
		hidden_input = numpy.dot(self.wih, inputs)
		hidden_output = self.actication_function(hidden_input)

		final_input = numpy.dot(self.who, hidden_output)
		final_output = self.actication_function(final_input)

		return final_output

if __name__ == "__main__":
	
	# 创建神经网络
	input_nodes = 28*28
	hidden_node =100
	output_nodes = 10
	learning_rate = 0.1

	net = neuralNetwork(input_nodes, hidden_node, output_nodes, learning_rate)

	# 训练神经网络
	data_file = open("./mnist_train.csv", 'r')
	data_list = data_file.readlines()
	data_file.close()

	# time表示重复训练的次数，每次训练的初始值不同，
	time = 1
	data_len = len(data_list)
	for i in range(time):
		for index, record in enumerate(data_list): 
			if index % 500 == 0:
				print("training ... %d%%" %(index/data_len * 100))
			all_valuse = record.split(',')
			#image_array = numpy.asfarray(all_valuse[1:]).reshape((28,28))
			#matplotlib.pyplot.imshow(image_array, cmap="Greys", interpolation='None')
			inputs = (numpy.asfarray(all_valuse[1:]) / 255.0 * 0.99) + 0.01
			traget = numpy.zeros(output_nodes) + 0.01
			traget[int(all_valuse[0])] = 0.99
			net.train(inputs, traget)
		pass
	pass


	# 使用训练后的网络进行推理
	test_data_file = open("./mnist_test.csv", 'r')
	test_data_list = test_data_file.readlines()
	test_data_file.close()

	# 记录测试对比结果
	scoredcard = []

	for record in test_data_list: 
		all_valuse = record.split(',')
		correct_lable = int(record[0])

		inputs = (numpy.asfarray(all_valuse[1:])/255.0 *0.99) +0.01
		outputs = net.query(inputs)

		output_label = numpy.argmax(outputs)
		# print("input/output is  %d/%d" % (correct_lable, output_label) )

		if (correct_lable == output_label):
			scoredcard.append(1)
		else:
			scoredcard.append(0)
		pass
	pass

	# 打印正确率
	scorecard_array = numpy.asarray(scoredcard)
	print ("performance = ", scorecard_array.sum() / scorecard_array.size)