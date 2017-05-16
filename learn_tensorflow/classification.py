# coding:utf8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

def add_layer(inputs, in_size, out_size, activation_function=None):
    #Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
    #biases = tf.Variable(tf.zeros([out_size]) + 0.1, name='b')
    Weights = get_weight(in_size, out_size)
    biases = get_biases(out_size)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def get_weight(in_size, out_size):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
    return Weights

def get_biases(out_size):
    biases = tf.Variable(tf.zeros([out_size]) + 0.1, name='b')
    return biases

def compute_accuracy(v_xs,v_ys):
    global prediction # 全局变量
    y_pre = sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1)) # 对比预测和真实数据的差别
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) # 计算准确率
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result

xs = tf.placeholder(tf.float32,[None,784]) # 不规定多少个sample但是有784个属性
ys = tf.placeholder(tf.float32,[None,10]) # 不规定多少个sample但是有10个属性


prediction = add_layer(xs,784,10,activation_function=tf.nn.softmax)
# 实验发现只要是l1就肯定是0，这他妈到底为啥！！！
# l1 = add_layer(xs,784,100,activation_function=None)
# prediction = add_layer(l1,100,10,activation_function=tf.nn.softmax)
# Weights = get_weight(784, 100)
# biases = get_biases(100)
# Wx_plus_b = tf.matmul(xs, Weights) + biases



cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=1)) #loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i % 50 == 0:
        #print sess.run(Wx_plus_b,feed_dict={xs:batch_xs,ys:batch_ys})
        #print sess.run(prediction,feed_dict={xs:batch_xs,ys:batch_ys})
        #print "\n==================================\n"
        print compute_accuracy(mnist.test.images,mnist.test.labels)