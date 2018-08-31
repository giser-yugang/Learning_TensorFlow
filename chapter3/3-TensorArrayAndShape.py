#-*- coding:utf-8 -*-
# @author: yugang
# @email: giser_yugang@163.com
# @datetime: 2018/8/28 10:55
# @software: PyCharm

import numpy as np
import tensorflow as tf

c = tf.constant([[1,2,3],[4,5,6]])
print('Python List input:{}'.format(c.get_shape()))

c = tf.constant(np.array([
    [[1,2,3],[4,5,6]],
    [[1,1,1],[2,2,2]]
]))
print('3d Numpy array input:{}'.format(c.get_shape()))

sess = tf.InteractiveSession()
c = tf.linspace(0.0,4.0,5)
print('The content of "c":\n {}\n'.format(c.eval()))
sess.close()

A = tf.constant([[1,2,3],[4,5,6]])
print(A.get_shape())

x = tf.constant([[1,0,1],[1,0,1]])
print(x.get_shape())
x = tf.expand_dims(x,1)
print('*')
print(x.get_shape())
print(x)

b = tf.matmul(A,x)
sess = tf.InteractiveSession()
print('matmul results:\n {}'.format(b.eval()))
sess.close()

#Name
with tf.Graph().as_default():
    c1 = tf.constant(4,dtype=tf.float64,name='c')
    c2 = tf.constant(4,dtype=tf.int32,name='c')
print(c1.name)
print(c2.name)
print()

#Name Scopes
with tf.Graph().as_default():
    c1 = tf.constant(4,dtype=tf.float64,name='c')
    with tf.name_scope('prefix_name'):
        c2 = tf.constant(4,dtype=tf.int32,name='c')
        c3 = tf.constant(4,dtype=tf.float64,name='c')
print(c1.name)
print(c2.name)
print(c3.name)