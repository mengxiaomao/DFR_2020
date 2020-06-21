# -*- coding: utf-8 -*-
"""
Created on Mon May 25 09:27:05 2020

@author: mengxiaomao
"""
import time
import h5py
import numpy as np
import scipy.io as sc
import tensorflow as tf
reuse = tf.AUTO_REUSE
dtype = np.float32
eps = 1e-4

N_pre = 1
N_suf = 2
class_num = 4
signal_size = 128
npacket = 500

max_iteration = 5
max_episode = 500
share_flag = True

data_file = 'DFR_1.mat'
weight_file = './Datasets/'

N_f = N_pre + N_suf
input_size = signal_size + class_num * N_f

def variable_w(shape):  
    w = tf.get_variable('w', shape = shape, initializer = tf.truncated_normal_initializer(stddev=0.1))
    return w
    
def variable_b(shape, initial = 0.01):
    b = tf.get_variable('b', shape = shape, initializer = tf.constant_initializer(initial))    
    return b  

def concat(z, p):    
    pre = tf.concat((tf.ones((N_pre, 1), dtype = dtype), tf.zeros((N_pre, class_num - 1), dtype = dtype)), axis = 1)
    suf = tf.concat((tf.ones((N_suf, 1), dtype = dtype), tf.zeros((N_suf, class_num - 1), dtype = dtype)), axis = 1)
    memory = tf.concat((pre, p, suf), axis = 0)
    q = tf.zeros((p.shape[0], 0), dtype = dtype)
    for i in range(N_f + 1):
        if i == N_pre:
            continue
        q = tf.concat((q, memory[i:i+npacket,:]), axis = 1)
    x = tf.concat((z, q), axis = 1)
    return x
    
#def NN_module(x, name):
#    hidden_size = 2
#    with tf.variable_scope(name + '.0', reuse = reuse):
#        w = variable_w([input_size, hidden_size])
#        b = variable_b([hidden_size])
#        l = tf.matmul(x, w) + b
#        l = tf.nn.relu(l)
#    with tf.variable_scope(name + '.1', reuse = reuse):
#        w = variable_w([hidden_size, class_num])
#        b = variable_b([class_num])
#        l = tf.matmul(l, w) + b
#        y_hat = tf.nn.softmax(l)
#    return y_hat
    
def NN_module(x, name):
    hidden_size = [128, 128]
    with tf.variable_scope(name + '.0', reuse = reuse):
        w = variable_w([input_size, hidden_size[0]])
        b = variable_b([hidden_size[0]])
        l = tf.matmul(x, w) + b
        l = tf.nn.relu(l)
    with tf.variable_scope(name + '.1', reuse = reuse):
        w = variable_w([hidden_size[0], hidden_size[1]])
        b = variable_b([hidden_size[1]])
        l = tf.matmul(l, w) + b
        l = tf.nn.relu(l)
    with tf.variable_scope(name + '.2', reuse = reuse):
        w = variable_w([hidden_size[1], class_num])
        b = variable_b([class_num])
        l = tf.matmul(l, w) + b
        y_hat = tf.nn.softmax(l)
    return y_hat
    
def DFR(z):
    y_hat = []
    p = tf.ones((z.shape[0], class_num), dtype = dtype) / class_num
    for k in range(max_iteration):
        x = concat(z, p)
        if share_flag:
            name = str(0)
        else:
            name = str(k)
        p = NN_module(x, name)                
        y_hat.append(p)
    return y_hat
    
def cost(y, y_hat):
    cost = tf.zeros((1,), dtype = dtype)
    for k in range(max_iteration):
        y_hat[k] = tf.clip_by_value(y_hat[k], 0, 1 - eps)
        loss = - tf.reduce_mean(y * tf.log(y_hat[k]) + (1 - y) * tf.log(1 - y_hat[k]))
        cost += loss
    return cost

def optimizer(cost): 
    with tf.variable_scope('opt', reuse = reuse):
        opt = tf.train.AdamOptimizer().minimize(cost)
    for var in tf.trainable_variables():
        print(var.name, var.shape)  
    return opt
  
def calculate_SER(y, y_hat):
    SER = []
    y_label = tf.argmax(y, axis = 1)
    for k in range(max_iteration):
        y_hat_label = tf.argmax(y_hat[k], axis = 1)
        same = tf.cast(tf.equal(y_label, y_hat_label), dtype = dtype)
        SER.append(1. - tf.reduce_mean(same))
    return SER
    
def save():
    dict_name={}
    for var in tf.trainable_variables(): 
        dict_name[var.name]=var.eval()
    sc.savemat(weight_file, dict_name)  
    
def NN_ini(theta):
    update=[]
    for var in tf.trainable_variables():
        update.append(tf.assign(tf.get_default_graph().get_tensor_by_name(var.name),tf.constant(np.reshape(theta[var.name],var.shape))))
    return update

def load_train_data():
    base_train_filepath = data_file + 'train/train_data_20MHz.mat'        
    train_data = np.array(sc.loadmat(base_train_filepath)['train_data'])
    train_x = train_data[:,:signal_size]
    train_y = train_data[:,signal_size:]  
    train_data=[] 
    return train_x, train_y
    
def load_valid_data():
    base_train_filepath = data_file + 'train/valid_data_20MHz.mat'        
    valid_data = np.array(sc.loadmat(base_train_filepath)['train_data'])
    valid_x = valid_data[:,:signal_size]
    valid_y = valid_data[:,signal_size:]  
    valid_data=[] 
    return valid_x, valid_y
    
def load_test_data(snr):
    base_test_filepath = data_file+'test/test_data_20MHz_'+str(snr)+'dB.mat'
    test_data=np.array(h5py.File(base_test_filepath)['train_data']).T
    test_x=test_data[:,:signal_size]
    test_y=test_data[:,signal_size:]
    test_data=[]
    return test_x, test_y
    
def train():
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        train_x, train_y = load_train_data()
        valid_x, valid_y = load_valid_data()
        best_valid_cost = 1000
        loss_valid_episode = list()
        loss_episode = list()
        SER_episode = list()
        for k in range(max_episode):
            st = time.time()
            loss_valid_batch = list()
            loss_batch = list()
            SER_batch = list()
            for i in range(train_x.shape[0] // npacket):
                batch_x = train_x[i*npacket:(i+1)*npacket][:]
                batch_y = train_y[i*npacket:(i+1)*npacket][:]
                                  
#                batch_y = np.clip(batch_y, 0, 1 - eps)
                _, cost_, SER_ = sess.run([opt, cost, SER], feed_dict = {z: batch_x, y: batch_y})
                loss_batch.append(cost_)
                SER_batch.append(SER_)
                
            for i in range(valid_x.shape[0] // npacket):
                batch_x = valid_x[i*npacket:(i+1)*npacket][:]
                batch_y = valid_y[i*npacket:(i+1)*npacket][:]
                cost_ = sess.run(cost, feed_dict = {z: batch_x, y: batch_y})
                loss_valid_batch.append(cost_)
            loss_valid_episode.append(np.mean(np.array(loss_valid_batch, dtype = dtype), axis = 0))
            loss_episode.append(np.mean(np.array(loss_batch, dtype = dtype), axis = 0))
            SER_episode.append(np.mean(np.array(SER_batch, dtype = dtype), axis = 0))
            if loss_valid_episode[-1] < best_valid_cost:
                save()
                best_valid_cost = loss_valid_episode[-1]
            elif np.isnan(loss_episode[-1]):
                print('error')
                break
            print("Episode(train):%d  Train Cost: %.3f  Valid Cost: %.3f  Time cost: %.2fs" 
                  %(k, loss_episode[-1],loss_valid_episode[-1], time.time()-st))
#            print(SER_episode[-1])             
 
def test(SNR):
    npacket = 5000
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        sess.run(load)
        SER_snr = list()
        for snr in SNR:        
            test_x, test_y = load_test_data(snr)
            SER_batch = list()
            for i in range(test_x.shape[0] // npacket):
                batch_x = test_x[i*npacket:(i+1)*npacket][:]
                batch_y = test_y[i*npacket:(i+1)*npacket][:]
                SER_ = sess.run(SER, feed_dict = {z: batch_x, y: batch_y})
                SER_batch.append(SER_)
            SER_snr.append(np.mean(np.array(SER_batch, dtype = dtype), axis = 0))
        print('\ntesting process is finished.')
    return np.array(SER_snr, dtype = dtype)
     
if __name__ == '__main__':  
    tf.reset_default_graph()
    z = tf.placeholder(shape = [npacket, signal_size], dtype = dtype)
    y = tf.placeholder(shape = [npacket, class_num], dtype = dtype)  
    y_hat = DFR(z)
    cost = cost(y, y_hat)
    opt = optimizer(cost)
    SER = calculate_SER(y, y_hat)

    train()
    
    npacket = 5000
    tf.reset_default_graph()
    z = tf.placeholder(shape = [npacket, signal_size], dtype = dtype)
    y = tf.placeholder(shape = [npacket, class_num], dtype = dtype)  
    y_hat = DFR(z)
    SER = calculate_SER(y, y_hat)
    load = NN_ini(sc.loadmat(data_file))  
    snr = [0,2,4,6,8,10]    
    SER_snr = test(snr)
    print(SER_snr)

        
