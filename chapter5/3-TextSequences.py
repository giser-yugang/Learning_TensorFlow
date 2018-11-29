#-*- coding:utf-8 -*-
# @author: yugang
# @email: giser_yugang@163.com
# @datetime: 2018/9/3 21:28
# @software: PyCharm

import tensorflow as tf
import numpy as np

def get_sentence_batch(batch_size,data_x,data_y,data_seqlens):
    instance_indices = list(range(len(data_x)))
    np.random.shuffle((instance_indices))
    batch = instance_indices[:batch_size]
    x = [[word2index_map[word] for word in data_x[i].lower().split()] for i in batch]
    y = [data_y[i] for i in batch]
    seqlens = [data_seqlens[i] for i in batch]
    return x,y,seqlens


if __name__ == '__main__':
    batch_size = 28
    embedding_dimmension = 64
    num_class=2
    hidden_layer_size=32
    times_step = 6
    element_size = 1


    digit_to_word_map = {1:'One',2:'Two',3:'Three',4:'Four',5:'Five'
                         ,6:'Six',7:'Seven',8:'Eight',9:'Nine'}
    digit_to_word_map[0] = 'PAD'

    even_sentences = []
    odd_sentences = []
    seqlens = []
    for i in range(10000):
        rand_seq_len = np.random.choice(range(3,7))
        seqlens.append(rand_seq_len)
        rand_odd_ints = np.random.choice(range(1,10,2),rand_seq_len)
        rand_even_ints = np.random.choice(range(2,10,2),rand_seq_len)
        #padding
        if rand_seq_len <6:
            rand_odd_ints = np.append(rand_odd_ints,[0]*(6-rand_seq_len))
            rand_even_ints = np.append(rand_even_ints,[0]*(6-rand_seq_len))

        even_sentences.append(' '.join([digit_to_word_map[r] for r in rand_odd_ints]))
        odd_sentences.append(' '.join([digit_to_word_map[r] for r in rand_even_ints]))

    data = even_sentences + odd_sentences
    seqlens *=2
    # print(even_sentences[:6])

    #map from words to indices
    word2index_map = {}
    index = 0
    for sent in data:
        for word in sent.lower().split():
            if word not in word2index_map:
                word2index_map[word] = index
                index += 1
    #inverse map
    index2word_map = {index:word for word,index in word2index_map.items()}
    vocabulary_size = len(index2word_map)

    # create labels and one-hot
    labels = [1]*10000+[0]*10000
    for i in range(len(labels)):
        label  =labels[i]
        one_hot_encoding = [0]*2
        one_hot_encoding[label] = 1
        labels[i] = one_hot_encoding

    #split train and test
    data_indices = list(range(len(data)))
    np.random.shuffle(data_indices)
    data = np.array(data)[data_indices]
    labels = np.array(labels)[data_indices]
    seqlens = np.array(seqlens)[data_indices]

    train_x = data[:10000]
    train_y = labels[:10000]
    train_seqlens = seqlens[:10000]

    test_x = data[10000:]
    test_y = labels[10000:]
    test_seqlens = seqlens[10000:]

    #start
    _inputs = tf.placeholder(tf.int32,shape=[batch_size,times_step])
    _labels = tf.placeholder(tf.float32,shape=[batch_size,num_class])
    #seqlens for dynamic calculation
    _seqlens = tf.placeholder(tf.int32,shape=[batch_size])

    #embedding
    with tf.name_scope('embeddings'):
        embedding = tf.Variable(tf.random_uniform([vocabulary_size,embedding_dimmension]
                                                  ,-1.0,1.0),name='embedding')
        embed = tf.nn.embedding_lookup(embedding,_inputs)
    #lstm
    num_LSTM_layer = 2
    with tf.variable_scope('lstm'):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_layer_size,forget_bias= 1.0)
        # lstm_cell_list = [tf.contrib.rnn.BasicLSTMCell(hidden_layer_size,forget_bias= 1.0) for _ in range(num_LSTM_layer)]
        # cell = tf.contrib.rnn.MultiRNNCell(cells=lstm_cell_list, state_is_tuple=True)
        outputs,states = tf.nn.dynamic_rnn(lstm_cell,embed,sequence_length=_seqlens,dtype=tf.float32)
    weights = {
        'linear_layer':tf.Variable(tf.truncated_normal([hidden_layer_size,num_class],mean=0,stddev=.01))
    }
    biases = {
        'linear_layer':tf.Variable(tf.truncated_normal([num_class],mean=0,stddev=.01))
    }

    final_output = tf.matmul(states[1], weights['linear_layer']) + biases['linear_layer']
    # final_output = tf.matmul(states[num_LSTM_layer-1][1],weights['linear_layer'])+biases['linear_layer']
    softmax = tf.nn.softmax_cross_entropy_with_logits(logits=final_output,labels=_labels)
    cross_entropy = tf.reduce_mean(softmax)

    #
    train_step = tf.train.RMSPropOptimizer(0.001,0.9).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(_labels,1),tf.argmax(final_output,1))
    accuracy = (tf.reduce_mean(tf.cast(correct_prediction,tf.float32)))*100
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(1000):
            x_batch,y_batch,seqlen_batch =get_sentence_batch(
                batch_size,train_x,train_y,train_seqlens)
            sess.run(train_step,feed_dict={_inputs:x_batch,_labels:y_batch,_seqlens:seqlen_batch})
            if step%100==0:
                acc = sess.run(accuracy,feed_dict={_inputs:x_batch,
                                                    _labels:y_batch,_seqlens:seqlen_batch})

                print('Accuracy at %d: %.5f'%(step,acc))
        for test_batch in range(5):
            x_test,y_test,seqlen_test = get_sentence_batch(
                batch_size,test_x,test_y,test_seqlens)
            batch_pred,batch_acc = sess.run([tf.argmax(final_output,1),accuracy]
                                            ,feed_dict={_inputs:x_test,
                                                        _labels:y_test,_seqlens:seqlen_test})
            print('Test batch accuracy %d: %.5f'%(test_batch,batch_acc))

        output_example = sess.run([outputs],feed_dict={_inputs:x_test,
                                                       _labels:y_test,_seqlens:seqlen_test})
        states_example = sess.run([states],feed_dict={_inputs:x_test,
                                                       _labels:y_test,_seqlens:seqlen_test})

        print(seqlen_test[1])
        print(output_example[0][1].shape)
        # print(output_example[0][1][:6,:3])
        print(output_example[0][1,-1, :])
        print(len(states_example[0]))
        print(states_example[0][1][1,:])
        # print(states_example[0][1][:3])
