#-*- coding:utf-8 -*-
# @author: yugang
# @email: giser_yugang@163.com
# @datetime: 2018/9/4 21:13
# @software: PyCharm

import tensorflow as tf
import numpy as np

class Model:
    def __init__(self):
        #model
        self.x = tf.placeholder(tf.float32,shape=[None,3])
        self.y_true = tf.placeholder(tf.float32,shape=None)
        self.w = tf.Variable([[0,0,0]],dtype=tf.float32)
        self.b = tf.Variable(0,dtype=tf.float32)

        init =tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

        self._output =None
        self._optimizer = None
        self._loss=None

    def fit(self,x_data,y_data):
        print(self.b.name)
        for step in range(10):
            self.sess.run(self.optimizer,{self.x:x_data,self.y_true:y_data})
            if (step%5==4)or (step==0):
                print(step,self.sess.run([self.w,self.b]))
    @property
    def output(self):
        if not self._output:
            y_pred = tf.matmul(self.w,tf.transpose(self.x))+self.b
            self._output = y_pred
        return self._output

    @property
    def loss(self):
        if not self._loss:
            error = tf.reduce_mean(tf.square(self.y_true-self.output))
            self._loss = error
        return self._loss

    @property
    def optimizer(self):
        if not self._optimizer:
            opt = tf.train.GradientDescentOptimizer(0.5)
            opt = opt.minimize(self.loss)
            self._optimizer=opt
        return self._optimizer

if __name__ =='__main__':
    x_data = np.random.randn(2000, 3)
    w_real = [0.3, 0.5, 0.1]
    b_real = -0.2
    noise = np.random.randn(1, 2000) * 0.1
    y_data = np.matmul(w_real, x_data.T) + b_real + noise


    lin_reg = Model()
    lin_reg.fit(x_data,y_data)
    lin_reg.fit(x_data, y_data)