# coding:utf8
import tensorflow as tf
import numpy as np

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.mul(input1,input2)

with tf.Session() as sess:
    print (sess.run(output,feed_dict={input1:[7.],input2:[2.]})) # placeholder和feed_dict是绑定的

x_data = np.linspace(-1,1,10)[:,np.newaxis]
print x_data