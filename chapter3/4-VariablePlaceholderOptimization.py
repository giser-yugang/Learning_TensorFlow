#-*- coding:utf-8 -*-
# @author: yugang
# @email: giser_yugang@163.com
# @datetime: 2018/8/28 11:16
# @software: PyCharm

import tensorflow as tf
import numpy as np

#Variable
init_val = tf.random_normal((1,5),0,1)
var = tf.Variable(init_val,name='var')
print('pre run:\n{}'.format(var))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    post_var = sess.run(var)
print('\npost run:\n{}'.format(post_var))

#Placeholders
x_data = np.random.randn(5,10)
w_data = np.random.randn(10,1)

with tf.Graph().as_default():
    x = tf.placeholder(tf.float32,shape=(5,10))
    w = tf.placeholder(tf.float32,shape=(10,1))
    b = tf.fill((5,1),-1.)
    xw = tf.matmul(x,w)

    xwb = xw+b
    s = tf.reduce_mean(xwb)
    with tf.Session() as sess:
        outs = sess.run(s,feed_dict={x:x_data,w:w_data})
print('outs = {}'.format(outs))

#Optimization:linear regression
x_data = np.random.randn(2000,3)
w_real = [0.3,0.5,0.1]
b_real = -0.2
noise = np.random.randn(1,2000)*0.1
y_data = np.matmul(w_real,x_data.T)+b_real+noise


NUM_STEPS = 10
g = tf.Graph()
wb_ = []
with g.as_default():

    x = tf.placeholder(tf.float32,shape=[None,3])
    y_true = tf.placeholder(tf.float32,shape=None)
    with tf.name_scope('inference') as scope:
        w = tf.Variable([[0,0,0]],dtype=tf.float32,name='weight')
        b = tf.Variable(0,dtype=tf.float32,name='bias')
        y_pred = tf.matmul(w,tf.transpose(x))+b
    with tf.name_scope('loss') as scope:
        loss = tf.reduce_mean(tf.square(y_true-y_pred))
    with tf.name_scope('train') as scope:
        learning_rate = 0.5
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for step in range(NUM_STEPS):
            sess.run(train,{x:x_data,y_true:y_data})
            if step%5==0:
                print(step,sess.run([w,b]))
                wb_.append(sess.run([w,b]))
        print(10,sess.run([w,b]))
        print()

#logistic regression
N = 20000
def sigmoid(x):
    return 1/(1+np.exp(-x))
x_data = np.random.randn(N,3)
w_real = [0.3,0.5,0.1]
b_real = -0.2
wxb = np.matmul(w_real,x_data.T)+b_real

y_data_pre_noise = sigmoid(wxb)
y_data = np.random.binomial(1,y_data_pre_noise)

NUM_STEPS = 50

x = tf.placeholder(tf.float32,shape=[None,3])
y_true = tf.placeholder(tf.float32,shape=None)
with tf.name_scope('inference') as scope:
    w = tf.Variable([[0,0,0]],dtype=tf.float32,name='weight')
    b = tf.Variable(0,dtype=tf.float32,name='bias')
    y_pred = tf.matmul(w,tf.transpose(x))+b
with tf.name_scope('loss') as scope:
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,logits=y_pred)
    loss = tf.reduce_mean(loss)
with tf.name_scope('train') as scope:
    learning_rate = 0.5
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(loss)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(NUM_STEPS):
        sess.run(train,{x:x_data,y_true:y_data})
        [w_eval, b_eval] = sess.run([w,b])
        if step%5 == 0:
            print(step,[w_eval, b_eval])
            wb_.append([w_eval, b_eval])
    print(50,[w_eval, b_eval])

