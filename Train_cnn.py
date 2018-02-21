import numpy as np
import csv
import os
import cv2
from random import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import scipy
from matplotlib import pyplot as plt
import tensorflow as tf
import pickle

with open('Training_features.pickle','rb') as f:
    train_data=pickle.load(f)
train_data=np.array(train_data)
train_x=list(train_data[:,0][:-1000])
train_y=list(train_data[:,1][:-1000])
test_x=list(train_data[:,0][-1000:])
test_y=list(train_data[:,1][-1000:])

hm_epochs=15
learning_rate = 0.001
batch_size = 128

n_input = 4096
n_classes = 10 
dropout = 0.75 

x = tf.placeholder(tf.float32, [None, 4096])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) 

def conv2d(x, W, b, s=1):
    x = tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def conv_net(x, weights, biases, dropout):
    # Reshape input 
    x = tf.reshape(x, shape=[-1, 64, 64, 1])
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

weights = {
    # 5x5 conv, 1 input, 128 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 64])),
    # 5x5 conv, 128 inputs, 128 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 64, 128])),
    # fully connected, 16*16*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([16*16*64, 1024])),
    # 1024 inputs, 10 outputs 
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([64])),
    'bc2': tf.Variable(tf.random_normal([128])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

prediction = conv_net(x, weights, biases, keep_prob)
cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(hm_epochs):
        epoch_loss = 0
        i=0
        while i < len(train_x):
            start = i
            end = i+batch_size
            batch_x = np.array(train_x[start:end])
            batch_y = np.array(train_y[start:end])
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,y: batch_y})
            epoch_loss += c
            i+=batch_size
        print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))
    print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))

def next_batch(int size):
    start = _index_in_epoch
    _index_in_epoch += batch_size
    if _index_in_epoch > _num_examples:
      # Finished epoch
      _epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      _images = _images[perm]
      _labels = _labels[perm]
      # Start next epoch
      start = 0
      _index_in_epoch = batch_size
      assert batch_size <= _num_examples
    end = _index_in_epoch
    return train_x[start:end], train_y[start:end]


correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 1
    while step * batch_size < training_iters:
        batch_x, batch_y = next_batch(batch_size)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            loss, acc = sess.run([cost,accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_x,
                                      y: test_y,
                                      keep_prob: 1.}))

