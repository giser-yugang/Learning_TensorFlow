#-*- coding:utf-8 -*-
# @author: yugang
# @email: giser_yugang@163.com
# @create time: 2018/9/15 13:24
# @software: PyCharm

import zipfile
import numpy as np
import tensorflow as tf

def get_glove(path_to_glove,word2index_map):
    embedding_weights = {}
    count_all_words = 0
    with zipfile.ZipFile(path_to_glove)as z:
        with z.open('glove.8408.300d.txt') as f:
            for line in f:
                vals = line.split()
                word = str(vals[0].decode('utf-8'))
                if word in word2index_map:
                    print(word)
                    count_all_words+=1
                    coefs = np.array(vals[1:],dtype='float32')
                    coefs/=np.linalg.norm(coefs)
                    embedding_weights[word]=coefs
                if count_all_words==vocabulary_size-1:
                    break
    return embedding_weights

def  get_sentence_batch(batch_size,data_x,data_y,data_seqlens):
    instance_indeces = list(range(len(data_x)))
    np.random.shuffle(instance_indeces)
    batch = instance_indeces[:batch_size]
    x = [[word2index_map[word] for word in data_x[i].split()] for i in batch]
    y = [data_y[i] for i in batch]
    seqlens = [data_seqlens[i] for i in batch]
    return x,y,seqlens

if __name__ =='__main__':
    path_to_glove = 'path/to/glove/file'
    pre_trained = True
    glove_size = 300
    batch_size = 128
    embedding_dimension = 64
    num_class =2
    hidden_layer_size = 32
    time_steps = 6

    digit_to_word_map = {1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five'
        , 6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine'}
    digit_to_word_map[0] = 'PAD_token'
    even_sentences =[]
    odd_sentences = []
    seqlens = []
    for i in range(10000):
        rand_seq_len = np.random.choice(range(3,7))
        seqlens.append(rand_seq_len)
        rand_odd_ints = np.random.choice(range(1,10,2),rand_seq_len)
        rand_even_ints = np.random.choice(range(2,10,2),rand_seq_len)
        if rand_seq_len<6:
            rand_odd_ints = np.append(rand_odd_ints,[0]*(6-rand_seq_len))
            rand_even_ints = np.append(rand_even_ints,[0]*(6-rand_seq_len))
        even_sentences.append(' '.join([digit_to_word_map[r] for r in rand_even_ints]))
        odd_sentences.append(' '.join([digit_to_word_map[r] for r in rand_odd_ints]))
    data = even_sentences+odd_sentences
    seqlens*=2
    labels = [1]*10000+[0]*10000
    for i in range(len(labels)):
        label = labels[i]
        one_hot_encoding = [0]*2
        one_hot_encoding[label]=1
        labels[i] = one_hot_encoding

    word2index_map = {}
    index = 0
    for sent in data:
        for word in sent.split():
            if word not in word2index_map:
                word2index_map[word] = index
                index+=1

    index2word_map = {index:word for word,index in word2index_map.items()}
    vocabulary_size = len(index2word_map)

    word2bedding_dict = get_glove(path_to_glove,word2index_map)
    embedding_matrix = np.zeros((vocabulary_size,glove_size))
    for word,index in word2index_map.items():
        if not word =='PAD_TOKEN':
            word_embedding = word2bedding_dict[word]
            embedding_matrix[index,:]=word_embedding

    #create train,test
    data_indeces = list(range(len(data)))
    np.random.shuffle(data_indeces)
    data = np.array(data)[data_indeces]
    labels = np.array(labels)[data_indeces]
    seqlens = np.array(seqlens)[data_indeces]

    train_x = data[:10000]
    train_y = labels[:10000]
    train_seqlens = seqlens[:10000]
    test_x = data[10000:]
    test_y = labels[10000:]
    test_seqlens = seqlens[10000:]

    _inputs = tf.placeholder(tf.int32,shape=[batch_size,time_steps])
    embedding_placeholder = tf.placeholder(tf.float32,[vocabulary_size,glove_size])
    _labels = tf.placeholder(tf.float32,shape=[batch_size,num_class])
    _seqlens = tf.placeholder(tf.int32,shape=[batch_size])


    if pre_trained:
        embeddings = tf.Variable(tf.constant(0.0,shape = [vocabulary_size,glove_size])
                                 ,trainable=True)
        embedding_init = embeddings.assign(embedding_placeholder)
        embed = tf.nn.embedding_lookup(embeddings,_inputs)
    else:
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size,embedding_dimension],-1.0,1.0))
        embed = tf.nn.embedding_lookup(embeddings,_inputs)

    with tf.name_scope('biGRU'):
        with tf.VariableScope('forward'):
            gru_fw_cell = tf.contrib.rnn.GRUCell(hidden_layer_size)
            gru_fw_cell = tf.contrib.rnn.DropoutWrapper(gru_fw_cell)
        with tf.VariableScope('backwwar'):
            gru_bw_cell = tf.contrib.rnn.GRUCell(hidden_layer_size)
            gru_bw_cell = tf.contrib.rnn.DropoutWrapper(gru_bw_cell)

        outputs,states = tf.nn.bidirectional_dynamic_rnn(
            cell_bw=gru_bw_cell,cell_fw=gru_fw_cell,inputs=embed,
            sequence_length=_seqlens,dtype=tf.float32,scope='BiGRU')
    states = tf.concat(values=states,axis=1)

    weights = {'linear_layer':tf.Variable(tf.truncated_normal([2*hidden_layer_size,num_class],
                                                              mean=0,stddev=.01))}
    biases = {'linear_layer':tf.Variable(tf.truncated_normal([num_class],
                                                             mean=0,stddev=.01))}
    final_output = tf.matmul(states,weights['linear_layer'])+biases['linear_layer']
    softmax = tf.nn.softmax_cross_entropy_with_logits(logits=final_output,labels=_labels)
    cross_entropy = tf.reduce_mean(softmax)

    train_step = tf.train.RMSPropOptimizer(0.001,0.9).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(_labels,1),tf.argmax(final_output,1))
    accuracy = (tf.reduce_mean(tf.cast(correct_prediction,tf.float32)))*100

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(embedding_init,feed_dict={embedding_placeholder:embedding_matrix})

        for step in range(1000):
            x_batch,y_batch,seqlens_batch = get_sentence_batch(
                batch_size,train_x,train_y,train_seqlens)
            sess.run(train_step,feed_dict={_inputs:x_batch,_labels:y_batch,_seqlens:seqlens_batch})
            if step%100==0:
                acc = sess.run(accuracy,
                               feed_dict={_inputs:x_batch,_labels:y_batch,_seqlens:seqlens_batch})
                print('Accuracy ast %d: %.5f'%(step,acc))

        for test_batch in range(5):
            x_test,y_test,seqlen_test = get_sentence_batch(batch_size,test_x,test_y,test_seqlens)
            batch_pred,batch_acc = sess.run([tf.argmax(final_output,1),accuracy],
                                            feed_dict={_inputs:x_test,_labels:y_test,_seqlens:seqlen_test})
            print('Test batch accuracy %d: %.5f'%(test_batch,batch_acc))

