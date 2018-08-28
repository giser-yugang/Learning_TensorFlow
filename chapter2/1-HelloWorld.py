#-*- coding:utf-8 -*-
# @author: yugang
# @email: giser_yugang@163.com
# @datetime: 2018/8/27 21:04
# @software: PyCharm
import tensorflow as tf
print(tf.__version__)

h = tf.constant('Hello')
w = tf.constant('World')
hw = h+w

with tf.Session() as sess:
    ans = sess.run(hw)
print(ans)