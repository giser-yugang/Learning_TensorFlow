#-*- coding:utf-8 -*-
# @author: yugang
# @email: giser_yugang@163.com
# @datetime: 2018/8/28 19:43
# @software: PyCharm

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

def weight_variable(shape):
    inital = tf.truncated_normal(shape,stddev=0.1)
    return  tf.Variable(inital)
def bias_variable(shape):
    inital = tf.constant(0.1,shape=shape)
    return tf.Variable(inital)
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool_2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
def conv_layer(input,shape):
    W=weight_variable(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(conv2d(input,W)+b)
def full_layer(input,size):
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size,size])
    b = bias_variable([size])
    return tf.matmul(input,W)+b

if __name__ =='__main__':
    x = tf.placeholder(tf.float32,shape=[None,784])
    y_ = tf.placeholder(tf.float32,shape=[None,10])
    x_image = tf.reshape(x,[-1,28,28,1])
    conv1 = conv_layer(x_image,shape=[5,5,1,32])
    conv1_pool = max_pool_2(conv1)
    conv2 = conv_layer(conv1_pool,shape=[5,5,32,64])
    conv2_pool = max_pool_2(conv2)
    conv2_flat = tf.reshape(conv2_pool,[-1,7*7*64])
    full_1 = tf.nn.relu(full_layer(conv2_flat,1024))

    keeP_prob = tf.placeholder(tf.float32)
    full_1_drop = tf.nn.dropout(full_1,keep_prob=keeP_prob)

    y_conv = full_layer(full_1_drop,10)

    DATA_DIR = '../chapter2/tmp/data'
    STEPS = 5000
    mnist = input_data.read_data_sets(DATA_DIR,one_hot=True)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv,labels=y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(STEPS):
            batch = mnist.train.next_batch(50)
            if i%100==0:
                train_accuracy = sess.run(accuracy,feed_dict={x:batch[0],y_:batch[1],keeP_prob:1.0})
                print('step {}, training accuracy {}'.format(i,train_accuracy))
            sess.run(train_step,feed_dict={x:batch[0],y_:batch[1],keeP_prob:0.5})
        X = mnist.test.images.reshape(10,1000,784)
        Y = mnist.test.labels.reshape(10,1000,10)
        test_accuracy = np.mean(
            [sess.run(accuracy,feed_dict={x:X[i],y_:Y[i],keeP_prob:1.0}) for i in range(10)])
    print('test accuracy:{}'.format(test_accuracy))