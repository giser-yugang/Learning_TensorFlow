#-*- coding:utf-8 -*-
# @author: yugang
# @email: giser_yugang@163.com
# @datetime: 2018/8/28 10:04
# @software: PyCharm

import tensorflow as tf


a = tf.constant(5)
b = tf.constant(2)
c = tf.constant(3)
#figure3-2
d = tf.multiply(a,b)
e = tf.add(c,b)
f = tf.subtract(d,e)

sess = tf.Session()
outs = sess.run(f)
sess.close()
print('outs={}'.format(outs))

#figure3-3:1
c = tf.multiply(a,b)
d = tf.add(a,b)
e = tf.subtract(c,d)
f = tf.add(c,d)
g = tf.div(f,e)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ans = sess.run(g)
print(ans)

#figure3-3:2
c = tf.multiply(a,b)
c= tf.cast(c,tf.float32)
d = tf.sin(c)
b= tf.cast(b,tf.float32)
e = tf.div(b,d)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    e = sess.run(e)
print(type(e))