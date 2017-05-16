# coding:utf-8
import tensorflow as tf
import numpy as np

x_data = np.random.rand(100).astype(np.float32) # 生成100个随机数，类型是float32
y_data = x_data*0.1+0.3
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0)) # tf.Variable是用来生成tf变量,这里就是生成一个[-1,1]数
biases = tf.Variable(tf.zeros([1])) # zeros可以生成一个0

y = Weights*x_data+biases # 针对每个x算出y值

loss = tf.reduce_mean(tf.square(y-y_data)) # 计算loss 平方求平均

optimizer = tf.train.GradientDescentOptimizer(0.5) # 找个训练方法就是什么梯度下降之类的
train = optimizer.minimize(loss) #　训练，让loss最小

init = tf.initialize_all_variables() # 初始化所有的变量

sess = tf.Session() # 会话，贼重要
sess.run(init) # 啥也得run才可以

print sess.run(Weights)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print step,sess.run(Weights),sess.run(biases)