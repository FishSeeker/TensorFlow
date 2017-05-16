# coding:utf8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs,in_size,out_size,n_layer,activation_function=None):
    layer_name = 'layer%s'% n_layer
    with tf.name_scope(layer_name): # 意思是这个with里的元素被包括在scope里
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W') # 就是建造一个数组，[行,列]
            tf.histogram_summary(layer_name+'/weights',Weights) # 让weights
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size]) + 0.1,name='b')
            tf.histogram_summary(layer_name+'/biases',biases)
        with tf.name_scope('Wx'):
            Wx_plus_b=tf.matmul(inputs,Weights) + biases # 计算一波
        if activation_function is None: # 如果不指定激励函数
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b) # 如果指定了激励函数
        tf.histogram_summary(layer_name+'/outputs',outputs)
        return outputs

x_data = np.linspace(-1,1,300)[:,np.newaxis] # 这个能生成300个数，后面中括号可以让它变成竖着的
noise = np.random.normal(0,0.05,x_data.shape) # 生成噪声点，这里是个正太分布，均值是0方差是0.05，和x_data一样的格式
y_data = np.square(x_data)-0.5+noise # 计算y

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,1],name='x_input')
    ys = tf.placeholder(tf.float32,[None,1],name='y_input')


l1 = add_layer(xs,1,10,n_layer=1,activation_function=tf.nn.relu) # 输入数据是xs,１个输入，１０个输出，名字是１，激励函数是relu
predition = add_layer(l1,10,1,n_layer=2,activation_function=None) # 输入数据是l1的输出,１０个输入，１个输出，名字是２，没激励函数
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-predition),reduction_indices=[1]))
    tf.scalar_summary('loss',loss)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("logs/",sess.graph) # 将所有的图画到log里
sess.run(init)

fig = plt.figure() # 用这个画图
ax = fig.add_subplot(1,1,1) # 图片编号
ax.scatter(x_data,y_data) # 点的编号
plt.ion() # 不用停顿
plt.show() #　输出图片


for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50 == 0:
        result = sess.run(merged,feed_dict={xs:x_data,ys:y_data})
        writer.add_summary(result,i) # 每i步输出一个结果
        try:
            ax.lines.remove(lines[0]) # 去除上次的线
        except Exception:
            pass
        #print sess.run(loss,feed_dict={xs:x_data,ys:y_data})
        predition_value = sess.run(predition,feed_dict={xs:x_data}) # 预测值
        lines = ax.plot(x_data,predition_value,'r-',lw=5) # 用红色，宽度为５的曲线的形式画出来预测曲线
        plt.pause(0.1)
