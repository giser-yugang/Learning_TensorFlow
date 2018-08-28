#-*- coding:utf-8 -*-
# @author: yugang
# @email: giser_yugang@163.com
# @datetime: 2018/8/28 10:33
# @software: PyCharm

import tensorflow as tf
print(tf.get_default_graph())

g = tf.Graph()
print(g)

a = tf.constant(5)
print(a.graph is g)
print(a.graph is tf.get_default_graph())
print()

#with statement
g1 = tf.get_default_graph()
g2 = tf.Graph()

print(g1 is tf.get_default_graph())
with g2.as_default():
    print(g1 is tf.get_default_graph())
print(g1 is tf.get_default_graph())
print()
#fetches
a = tf.constant(5)
b = tf.constant(2)
c = tf.constant(3)
d = tf.constant(10)
e = tf.constant(5)
f = tf.constant(5)

with tf.Session() as sess:
    fetchs = [a,b,c,d,e,f]
    outs = sess.run(fetchs)
print('outs = {}'.format(outs))
print(type(outs[0]))

