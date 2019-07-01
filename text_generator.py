
#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
import numpy as np
import nltk


class rnn_model:
    def __init__(self,glove_vec):
        self.word_ids = tf.placeholder(tf.int32, shape=[None,None])
        self.l_outputs = tf.placeholder(tf.int32,[None,None])
        self.glove = tf.Variable(glove_vec, dtype=tf.float32, trainable=False)
        self.word_embedded = tf.nn.embedding_lookup(self.glove,self.word_ids)
        self.rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=50)
        self.outputs, self.state = tf.nn.dynamic_rnn(cell=self.rnn_cell, dtype = tf.float32, inputs=self.word_embedded,time_major = False)
        self.projections = tf.layers.dense(self.outputs,len(glove_vec), activation=tf.nn.relu)
        self.error = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.l_outputs,logits=self.projections))
        self.opt = tf.train.AdamOptimizer()
        self.optimizer = self.opt.minimize(self.error)
        self.acc, self.acc_op = tf.metrics.accuracy(labels=self.l_outputs, predictions=tf.argmax(self.projections,2))



def read_data(data):
    with open(data, "r") as f:
        sents = f.read()
    tokens = nltk.word_tokenize(sents)
    tokens = [t.lower() for t in tokens]
    return tokens



def batching(tokens, batch_size, sequence_length):
    ipt = []
    labels = []
    n_sequence = (len(tokens)-1)//sequence_length
    for i in range(n_sequence):
        ipt.append(tokens[i*sequence_length:(i+1)*sequence_length])
        labels.append(tokens[i*sequence_length+1:(i+1)*sequence_length+1])
    total = len(ipt)
    nbatch = total//batch_size
    input_batches = []
    labels_batches = []
    for i in range(nbatch):
        input_batches.append(ipt[i*batch_size:(i+1)*batch_size])
        labels_batches.append(labels[i*batch_size:(i+1)*batch_size])
    return input_batches,labels_batches



def glove_l(file):
    lst = [np.random.rand(50).tolist()]
    with open(file) as f:
        cont = f.readlines()
    for i in range(len(cont)):
        line = cont[i].split()
        lst.append(list(map(float, line[1:])))
    return np.array(lst)



def wmap(file):
    words_map = dict()
    id2word = ["**UNKNOWN**"]
    words_map["**UNKNOWN**"] = 0
    with open(file) as f:
        cont = f.readlines()
    for i in range(len(cont)):
        line = cont[i].split()
        words_map[line[0]] = i+1
        id2word.append(line[0])
    return words_map,id2word



def preparedata(tokens,batch_size,len_seq,wordvec,words_map):
    tokens,labels = batching(tokens,batch_size,len_seq)
    input_tokens = []
    for batch in tokens:
        tmp_b = []
        for seq in batch:
            tmp = []
            for token in seq:
                if token in words_map:tmp.append(words_map[token])
                else:tmp.append(words_map["**UNKNOWN**"])
            tmp_b.append(tmp)
        input_tokens.append(tmp_b)
    input_labels = []
    for batch in labels:
        tmp_b = []
        for seq in batch:
            tmp = []
            for label in seq:
                if label in words_map:tmp.append(words_map[label])
                else:tmp.append(words_map["**UNKNOWN**"])
            tmp_b.append(tmp)
        input_labels.append(tmp_b)
    return input_tokens, input_labels



def batch_generator(word,label):
    nbatch = len(word)
    arr = np.arange(nbatch)
    np.random.shuffle(arr)
    train = arr[:nbatch//10*8]
    val = arr[nbatch//10*8:]
    train_word = [word[x] for x in train]
    train_label = [label[x] for x in train]
    val_word = [word[x] for x in val]
    val_label = [label[x] for x in val]
    return train_word,train_label,val_word,val_label



def generate_text(seedword,text_len,glove_vec,words_map,id2word):
    tf.reset_default_graph()
    sec_rnn = rnn_model(glove_vec)
    saver = tf.train.Saver()
    token = [[words_map[seedword]]]
    text = []
    with tf.Session() as sess:
        saver.restore(sess, "model.ckpt")
        print("Model restored.")
        for i in range(text_len):
            prj = sess.run(sec_rnn.projections,feed_dict={sec_rnn.word_ids:token})
            c = tf.argmax(prj,2)
            c = sess.run(c)
            text.append(id2word[c[0][0]])
            token = c
        print(text)



def main():
    batch_size,len_seq = 32, 50
    glove_vec = glove_l("glove.6B.50d.txt")
    words_map,id2word = wmap("glove.6B.50d.txt")
    tokens = read_data("al")
    input_tokens,input_labels = preparedata(tokens,batch_size,len_seq,glove_vec,words_map)
    rnn = rnn_model(glove_vec)
    iteration_per_epoch = 50
    tr_word, tr_opt, val_word, val_opt = batch_generator(input_tokens,input_labels)
    init = tf.initializers.global_variables()
    local_init = tf.local_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)
    sess.run(local_init)
    for epoch in range(100):
        epoch_error = 0
        for i in range(iteration_per_epoch):
            num = np.arange(len(tr_word))
            np.random.shuffle(num)
            word = tr_word[num[0]]
            opt = tr_opt[num[0]]
            epoch_error += sess.run([rnn.error,rnn.optimizer], feed_dict={rnn.word_ids:word,rnn.l_outputs:opt})[0]
        epoch_error /= iteration_per_epoch
        numval = np.arange(len(val_word))
        np.random.shuffle(numval)
        val_word_ids = val_word[numval[0]]
        val_opt_ids = val_opt[numval[0]]
        valid_accuracy = sess.run([rnn.acc, rnn.acc_op],{rnn.word_ids:val_word_ids,rnn.l_outputs:val_opt_ids})[1]
        print("Epoch %d, train error: %.2f, valid accuracy: %.1f %%" % (epoch, epoch_error, valid_accuracy * 100.0))
    save_path = saver.save(sess, "model.ckpt")
    generate_text("when",13,glove_vec,words_map,id2word)



main()
