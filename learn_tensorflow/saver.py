import tensorflow as tf
import numpy as np

## save to file
# remember to define the same dtype and shape when restore
W = tf.Variable([[1,2,3],[3,4,5]],dtype=tf.float32,name='weights')
b = tf.Variable([[1,2,3]],dtype=tf.float32,name='biases')

init = tf.initialize_all_variables()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess,"/home/fish/workspace/spark_compile/learn_tensorflow/my_net/save_net.ckpt")
    print "save to path:"+save_path

# restor variable cannot save net 
# redifine the same shape and same type for your varibles
W = tf.Variable(np.arange(6).reshape((2,3)),dtype=tf.float32,name="weights")
b = tf.Variable(np.arange(3).reshape((1,3)),dtype=tf.float32,name="biases")

#not need init step
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,"/home/fish/workspace/spark_compile/learn_tensorflow/my_net/save_net.ckpt")

    print sess.run(W)
    print sess.run(b)