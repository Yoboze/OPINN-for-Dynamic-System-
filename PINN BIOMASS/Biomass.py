import numpy as np
import pandas as pd
import timeit
import time
import matplotlib.pyplot as plt
import random
import scipy.io

import tensorflow.compat.v1 as tf
from tqdm import tqdm
import timeit 
tf.disable_v2_behavior()

dat1 =pd.read_csv('data120.csv')
dat1.head()

z1_new = np.array(dat1["u1"]).flatten()[:,None]
z2_new = np.array(dat1["u2"]).flatten()[:,None]
z3_new = np.array(dat1["u3"]).flatten()[:,None]
t_new = np.array(dat1["t"]).flatten()[:,None]

from scipy import interpolate

# Assuming that the 't_data' array is the independent variable and 'z1_data' and 'z2_data' are dependent variables
t_data = np.linspace(t_new.min(), t_new.max(), 100)[:, None]  # generating 500 points

# Interpolation for z1_data
f_z1 = interpolate.interp1d(t_new.flatten(), z1_new.flatten(), kind='cubic')  # 'cubic' for cubic spline interpolation
z1_data = f_z1(t_data)

# Interpolation for z2_data
f_z2 = interpolate.interp1d(t_new.flatten(), z2_new.flatten(), kind='cubic')  # 'cubic' for cubic spline interpolation
z2_data = f_z2(t_data)

# Interpolation for z3_data
f_z3 = interpolate.interp1d(t_new.flatten(), z3_new.flatten(), kind='cubic')  # 'cubic' for cubic spline interpolation
z3_data = f_z3(t_data)

class PINN:
    # Initialize the class
    def __init__(self, t, z1, z2, z3, layers, layers1, layers2, layers3):
        
        self.lb = t.min(0)
        self.ub = t.max(0)
        
        self.t = t
        
        self.z1 = z1
        self.z2 = z2
        self.z3 = z3
        
        self.layers = layers
        self.layers1 = layers1
        self.layers2 = layers2
        self.layers3 = layers3
        
        self.weights, self.biases = self.initialize_NN(layers)
        self.weights1, self.biases1 = self.initialize_NN(layers1)
        self.weights2, self.biases2 = self.initialize_NN(layers2)
        self.weights3, self.biases3 = self.initialize_NN(layers3)
        
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        
        
        
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.z1_tf = tf.placeholder(tf.float32, shape=[None, self.z1.shape[1]])
        self.z2_tf = tf.placeholder(tf.float32, shape=[None, self.z2.shape[1]])
        self.z3_tf = tf.placeholder(tf.float32, shape=[None, self.z3.shape[1]])
        
        
        
        self.z1_pred, self.z2_pred, self.z3_pred = self.net_ASIR(self.t_tf)
        self.aa_pred = self.aa_net(self.t_tf)
        self.bb_pred = self.bb_net(self.t_tf)
        self.cc_pred = self.cc_net(self.t_tf)
        
        self.l1, self.l2, self.l3 = self.net_l(self.t_tf)
        
        
        self.loss = tf.reduce_sum(tf.square(self.z1_tf - self.z1_pred)) + \
                    tf.reduce_sum(tf.square(self.z2_tf - self.z2_pred)) + \
                    tf.reduce_sum(tf.square(self.z3_tf - self.z3_pred)) + \
                    tf.reduce_sum(tf.square(self.l1)) + \
                    tf.reduce_sum(tf.square(self.l2)) + \
                    tf.reduce_sum(tf.square(self.l3))
        
        # self.optimizer = tf.train.AdamOptimizer(1e-3)
        # self.train_op = self.optimizer.minimize(self.loss)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
        self.loss_log = []
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, t, layers, weights, biases):
        num_layers = len(layers1)
        
        H = 2.0*(t - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
        
    def neural_net1(self, t, layers1, weights1, biases1):
        num_layers = len(layers1)
        
        H = 2.0*(t - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights1[l]
            b = biases1[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights1[-1]
        b = biases1[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    def neural_net2(self, t, layers2, weights2, biases2):
        num_layers = len(layers1)
        
        H = 2.0*(t - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights2[l]
            b = biases2[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights2[-1]
        b = biases2[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    
    def neural_net3(self, t, layers3, weights3, biases3):
        num_layers = len(layers1)
        
        H = 2.0*(t - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights3[l]
            b = biases3[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights3[-1]
        b = biases3[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    
    def net_ASIR(self, t):
        ASIR = self.neural_net(t, self.layers, self.weights, self.biases)
        z1 = ASIR[:,0:1]
        z2 = ASIR[:,1:2]
        z3 = ASIR[:,2:3]
        
        return z1, z2, z3
    
    def aa_net(self,t):
        aa = self.neural_net1(t, self.layers1, self.weights1, self.biases1)
        alp = aa
        return alp
    
    def bb_net(self,t):
        bb = self.neural_net2(t, self.layers2, self.weights2, self.biases2)
        blp = bb
        return blp
    
    def cc_net(self,t):
        cc = self.neural_net2(t, self.layers3, self.weights3, self.biases3)
        clp = cc
        return clp
    
    
    
    def net_l(self, t):
        z1, z2, z3 = self.net_ASIR(t)
        aa = self.aa_net(t)
        bb = self.bb_net(t)
        cc = self.cc_net(t)
        
        z1_t = tf.gradients(z1, t)[0]
        z2_t = tf.gradients(z2, t)[0]
        z3_t = tf.gradients(z3, t)[0]
        
        


        
        l1 = z1_t - (-aa*z1 + bb*z2)
        l2 = z2_t - (-bb*z2 + cc*z3)
        l3 = z3_t - (-cc*z3)
        
        return l1, l2, l3
    
    
        
        
    def train(self, nIter):
        tf_dict = {self.t_tf: self.t, self.z1_tf: self.z1, self.z2_tf: self.z2, self.z3_tf: self.z3}
        start_time = timeit.default_timer()

        for it in tqdm(range(nIter)):
            self.sess.run(self.train_op, tf_dict)
            if it % 100 == 0:
                elapsed = timeit.default_timer() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                self.loss_log.append(loss_value)
                start_time = timeit.default_timer()
                
    def predict(self, t_star):
        tf_dict = {self.t_tf: t_star}
        
        z1_star = self.sess.run(self.z1_pred, tf_dict)
        z2_star = self.sess.run(self.z2_pred, tf_dict)
        z3_star = self.sess.run(self.z3_pred, tf_dict)
        aa_star = self.sess.run(self.aa_pred, tf_dict)
        bb_star = self.sess.run(self.bb_pred, tf_dict)
        cc_star = self.sess.run(self.cc_pred, tf_dict)
        
        return z1_star, z2_star, z3_star, aa_star, bb_star, cc_star
        
niter = 71800  # number of Epochs
layers = [1, 10, 3]
layers1 = [1, 10, 1]
layers2 = [1, 10, 1]
layers3 = [1, 10, 1]

model = PINN(t_data, z1_data, z2_data, z3_data, layers, layers1, layers2, layers3)
model.train(niter)

# prediction
z1_pred, z2_pred, z3_pred, aa_pred, bb_pred, cc_pred = model.predict(t_data)