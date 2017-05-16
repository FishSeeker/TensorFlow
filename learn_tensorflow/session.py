# coding:utf-8
import tensorflow as tf

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],
                       [2]])

product = tf.matmul(matrix1,matrix2)  # matrix multiply np.dot(m1,m2)

# method 1
sess = tf.Session()
result = sess.run(product)
print result
sess.close()

# method 2
with tf.Session() as sess: # 自动关
    result2 = sess.run(product)
    print result2

state = tf.Variable(0,name='counter')
one = tf.constant(1)
new_value = tf.add(state,one) # state+1的值赋给new_value
update = tf.assign(state,new_value) # 更新state的值

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3): # 更新３次
        sess.run(update)
        print sess.run(state)