# -*- coding: utf-8 -*-
"""
@author: Youngjoo Kim
@last modified: 12 Dec 2018
@Please cite the following paper if you find this code helpful:
    Youngjoo Kim et al., "A Capsule Network for Traffic Speed Prediction in Complex Road Networks",
    in Proceedings of Sensor Data Fusion: Trends, Solutions, and Applications (SDF), Bonn, Germany, Oct 2018.
"""

import math
import os
import time

import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from MinMaxScaler import MinMaxScaler

####
whichNet = 1 # 0 for CNN_Ma, others for CapsNet

####
drawPlot = 0 # draw plot if 1
printEvery = 100 # print loss every __ epoch

#### common settings

shuffle_batch = False    # shuffle image dataset 
batch_size = 1         # number of dataset (image) processed in one epoch
time_step_in = 10       # number of time steps of input (1 step = 15 min)
time_step_out = 1       # number of time steps of output (1 step = 15 min)

num_sensor_set = -1         # number of sensors (-1 for data of all sensors)
num_data_set = 5000        # number of data to use in train and eval (-1 for data of all time steps)
num_data_train_set = -1   # number of train data (-1 for default, 3/4) remaining data is used in evaluation
## load_data() will prepare data of M x N dimensions where M: num_data_set, N: num_sensor_set                   

#### CNN_Ma 
CNN_init_learning_rate = 0.0005 # starting learning rate
CNN_decay_rate = 0.9999 		# decay rate in exponential learning rate decay

CNN_initializer_stddev = 0.05   # weight initialize stddev
CNN_regularizer = 0.001         # l2 regularizer

## layer modification for CNN can be done in the function CNN_Ma()

#### CapsNet
CAP_init_learning_rate = 0.0005    # starting learning rate
CAP_decay_rate = 0.9999         # decay rate in exponential learning rate decay

CAP_initializer_stddev = 0.05   # weight initialize stddev
CAP_regularizer = 0.001        # l2 regularizer

CAP_routing_stddev = 0.05        # weight initialize stddev for routing layer
CAP_iter_routing = 3            # number of iterations of dynamic routing algorithm

CAP_conv1_num_filter = 32          # number of conv filters in the first layer
CAP_caps1_num_filter = 128          # numver of conv filters in the first capsule layer
CAP_caps1_vec_len = 8               # vector length of the first capsule layer
CAP_caps2_vec_len = 16              # vector length of the second capsule layer
repeatConv = True

#### file paths
checkpoint_dir = 'log'
checkpoint_path = checkpoint_dir+'/checkpoint.ckpt'
case = 'case1'
#case = 'case2'
eval_dir = 'eval/'+case
data_path = 'data/Santander_ST_speed_pp_'+case+'.csv'


## NO-tune global variables
num_channel = 1 # if RGB, it is 3. for speed-only data, we have 1 channel
num_data = -1
num_data_train = -1
num_data_eval = -1
num_sensor = 50
###############################################################################

dataset_train = np.empty
dataset_eval = np.empty

dataset_image_train = np.empty
dataset_image_eval = np.empty

# minmax scaler to convert speed data into a prespecified range
scaler = MinMaxScaler()

def main(_):
    
    load_data()
    train()
    evaluate()
    
    
# implementation of the proposed CapsNet based on [Sabour 2017]
def CapsNet(inputs):
    
    epsilon = 1e-9
    CAP_caps2_num_outputs = num_sensor*time_step_out # number of outputs in the second capsule layer
    
    print_activations(inputs) # 'Placeholder'
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0,CAP_initializer_stddev),
                        weights_regularizer=slim.l2_regularizer(CAP_regularizer)):
        
        # First conv layer
        conv1 = slim.conv2d(inputs, CAP_conv1_num_filter, [3, 3], scope='conv1')
        if repeatConv == True:
            conv1 = slim.conv2d(conv1, CAP_conv1_num_filter, [3, 3], scope='conv1/2')
        print_activations(conv1) # 'conv1/2/Relu'
        
        # PrimaryCaps layer
        caps1 = slim.conv2d(conv1, CAP_caps1_num_filter, [3, 3], scope='PrimaryCaps_layer')
        print_activations(caps1) # 'PrimaryCaps_layer/Relu'
        caps1 = tf.reshape(caps1, (batch_size, -1, CAP_caps1_vec_len, 1))
        print_activations(caps1) # 'Reshape'
        caps1 = squash(caps1)
        
        # TrafficCaps layer
        caps2 = tf.reshape(caps1, shape=(batch_size, -1, 1, caps1.shape[-2].value, 1))
        b_IJ = tf.constant(np.zeros([batch_size, caps2.shape[1].value, CAP_caps2_num_outputs, 1, 1], dtype=np.float32))
        caps2 = routing(caps2, b_IJ)
        caps2 = tf.squeeze(caps2, axis=1)
        print_activations(caps2) # 'Squeeze'
        
        caps2 = tf.sqrt(tf.reduce_sum(tf.square(caps2),
                                               axis=2, keepdims=True) + epsilon)

        # flattening
        flat = slim.flatten(caps2, scope='ft')
        print_activations(flat) # 'ft/flatten/Reshape'
       
    return flat

    
# implementation of [Ma 2017]
def CNN_Ma(inputs):

    print_activations(inputs)
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0,CNN_initializer_stddev),
                        weights_regularizer=slim.l2_regularizer(CNN_regularizer)):
        net = slim.conv2d(inputs, 256, [3, 3], scope='conv1')
        print_activations(net)
        
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        print_activations(net)
        
        net = slim.conv2d(net, 128, [3, 3], scope='conv2')
        print_activations(net)
        
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        print_activations(net)
        
        net = slim.conv2d(net, 64, [3, 3], scope='conv3')
        print_activations(net)
        
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        print_activations(net)
        
        net = slim.flatten(net, scope='ft')
        print_activations(net)
        net = slim.fully_connected(net, num_sensor*time_step_out*num_channel, activation_fn=None, scope='fc1')
        print_activations(net)
        
    return net

def print_num_parameters():
    
    print ("num_param:",
           np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]) )
    

def train():
  
    with tf.Graph().as_default():

        x_batch, y_batch = get_batch(0, train=True) # just for getting size   
      
        X = tf.placeholder(tf.float32, shape=x_batch.shape)
        Y = tf.placeholder(tf.float32, shape=y_batch.shape)

        if whichNet == 0: # if CNN_Ma
            predictions = CNN_Ma(X)
            init_learning_rate = CNN_init_learning_rate
            decay_rate = CNN_decay_rate
            print("<Training CNN_Ma>")
        else: # if CapsNet
            predictions = CapsNet(X)
            init_learning_rate = CAP_init_learning_rate
            decay_rate = CAP_decay_rate
            print("<Training CapsNet>")

        total_loss = tf.sqrt(tf.losses.mean_squared_error(labels=Y, predictions=predictions))

        step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(init_learning_rate, step, 1, decay_rate)       
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(total_loss, global_step=step)

        num_epoch = math.floor((num_data_train - (time_step_in + time_step_out + batch_size))/batch_size)
        print("Number of epoch:",num_epoch)

        loss_res = np.array(np.zeros(shape=[2, num_epoch]))
        start_time = time.time()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            print_num_parameters()
            sess.run(tf.global_variables_initializer())
            for epoch in range(num_epoch):
                x_batch, y_batch = get_batch(epoch, train=True)
                total_loss_out, _ = sess.run([total_loss, optimizer],
                                  feed_dict={X: x_batch,
                                             Y: y_batch})
                
                total_loss_out = scaler.scale_inverse(total_loss_out)
                
                loss_res[0, epoch] = epoch
                loss_res[1, epoch] = total_loss_out
                if epoch % printEvery == 0:
                    print("Epoch", epoch, " Loss:", total_loss_out)         
                
            duration = time.time()-start_time
            print("Training duration:", format(duration, ".3f"), "s, per epoch:", format(duration/num_epoch, ".3f"), "s")

            saver.save(sess, checkpoint_path)
            print("Model saved in path:", checkpoint_path)

        if drawPlot == 1:
            plt.xlabel('Epoch')
            plt.ylabel('Loss (RMSE)')
            plt.xlim([0, num_epoch])
            plt.ylim([0, 60])
            plt.plot(loss_res[0,:], loss_res[1,:])
            plt.grid(True)
            plt.title('Train')
            plt.show()
        else:
            print("Plot off") 
            
    
def evaluate():
 
    with tf.Graph().as_default():
                
        x_batch, y_batch = get_batch(0, train=False)

        X = tf.placeholder(tf.float32, shape=x_batch.shape)
        Y = tf.placeholder(tf.float32, shape=y_batch.shape)

        if whichNet == 0: # if CNN_Ma
            predictions = CNN_Ma(X)
            print("<Evaluating CNN_Ma>")
        else: # if CapsNet
            predictions = CapsNet(X)
            print("<Evaluating CapsNet>")
    	
        total_loss = tf.sqrt(tf.losses.mean_squared_error(labels=Y, predictions=predictions))

        saver = tf.train.Saver()
        
        num_epoch = math.floor((num_data_eval - (time_step_in + time_step_out + batch_size))/batch_size)
        loss_res = np.array(np.zeros(shape=[2, num_epoch]))
       
        # panda dataframe to store eval results
        df_error = pd.DataFrame()
        df_predicted = pd.DataFrame()       
        start_time = time.time()
        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
            for epoch in range(num_epoch):
                
                x_batch, y_batch = get_batch(epoch, train=False)

                truth, predicted, total_loss_out = sess.run([Y, predictions, total_loss],
                                            feed_dict={X: x_batch,
                                                       Y: y_batch})
                
                total_loss_out = scaler.scale_inverse(total_loss_out)
                truth = scaler.scale_inverse(truth)
                predicted = scaler.scale_inverse(predicted)
    
                loss_res[0, epoch] = epoch
                loss_res[1, epoch] = total_loss_out
                if epoch % printEvery == 0:
                    print("Epoch", epoch, " Loss:", total_loss_out)
                
                error = truth - predicted
                
                df_error = df_error.append(pd.DataFrame(error))
                df_predicted = df_predicted.append(pd.DataFrame(predicted))
                        
            duration = time.time()-start_time
            print("Evaluation duration:", format(duration, ".3f"), "s, per epoch:", format(duration/num_epoch, ".3f"), "s")
            
            if not os.path.exists(eval_dir):
                os.makedirs(eval_dir)
            
            if whichNet == 0:
                eval_error_path = eval_dir+'/result_error_CNN.csv'
                eval_prediction_path = eval_dir+'/result_prediction_CNN.csv'
                eval_RMSE_path = eval_dir+'/result_RMSE_CNN.csv'
            else:
                eval_error_path = eval_dir+'/result_error_CAP.csv'
                eval_prediction_path = eval_dir+'/result_prediction_CAP.csv'
                eval_RMSE_path = eval_dir+'/result_RMSE_CAP.csv'
            
            df_error.to_csv(eval_error_path, header=None, index=None)
            df_predicted.to_csv(eval_prediction_path, header=None, index=None)
            df_RMSE = pd.DataFrame(loss_res[1,:])
            df_RMSE.to_csv(eval_RMSE_path, header=None, index=None)
            print("Evaluation result saved in:", eval_dir)
            
        print(">>>Average Loss:",np.mean(loss_res[1,:]))
    
        ## plots
        if drawPlot == 1:
            plt.xlabel('Epoch')
            plt.ylabel('Loss (RMSE)')
            plt.xlim([0, num_epoch])
            plt.plot(loss_res[0,:], loss_res[1,:])
            plt.grid(True)
            plt.title('Evaluation')
            plt.show()
        else:
            print("Plot off")


def load_data():
    
    global num_sensor, num_data, num_data_train, num_data_eval, dataset_train, dataset_eval
    
    # read file
    dataset_read = pd.read_csv(data_path, header=None)
    
    num_data_read, num_sensor_read = dataset_read.shape
    if num_data_set == -1 or num_data_set > num_data_read:
        num_data = num_data_read
    else:
        num_data = num_data_set
    
    if num_sensor_set == -1 or num_sensor_set > num_sensor_read:
        num_sensor = num_sensor_read
    else:
        num_sensor = num_sensor_set
        
    dataset = dataset_read.iloc[:num_data, :num_sensor]
    assert num_data, num_sensor == dataset.shape
    
    dataset = np.array(dataset.values)
    
    scaler.fit(dataset, feature_range=(0,1))
    dataset = scaler.scale(dataset)
    
    time_per_day = 96 # 15 min x 96 time steps = 1 day
    print("Traffic data loaded from:", data_path)
    print("Number of data:", num_data, "=",math.ceil(num_data/time_per_day),"days  Number of sensors:", num_sensor)
    
    # split dataset
    if num_data_train_set == -1 or num_data_train_set > num_data:
        num_data_train = math.ceil(num_data*3.0/4.0)
    else:
        num_data_train = num_data_train_set 
        
    num_data_eval = num_data - num_data_train
    
    dataset_train = dataset[:num_data_train,:]
    dataset_eval = dataset[num_data_train:,:]
    print("First", num_data_train, "data => trainning set, the other", num_data_eval,"data => evaluation set")
    
    get_dataset_image(shuffle=shuffle_batch)
    

def get_dataset_image(shuffle=True):
    
    global dataset_image_train, dataset_image_eval
    
    num_image_train = num_data_train - (time_step_in + time_step_out) + 1
    num_image_eval = num_data_eval - (time_step_in + time_step_out) + 1
    
    dataset_image_train = np.array(np.zeros(shape=[num_image_train,
                                      time_step_in + time_step_out, # x and y together
                                      num_sensor,
                                      num_channel]))
    
    dataset_image_eval = np.array(np.zeros(shape=[num_image_eval,
                                      time_step_in + time_step_out, # x and y together
                                      num_sensor,
                                      num_channel]))
    
    for idx in range(num_image_train):
        idx_end = idx+time_step_in+time_step_out
        dataset_image_train[idx, :, :, 0] = dataset_train[idx:idx_end,:]
    
    for idx in range(num_image_eval):
        idx_end = idx+time_step_in+time_step_out
        dataset_image_eval[idx, :, :, 0] = dataset_eval[idx:idx_end,:]
        if (idx == 0):
            print("first idx_end of eval:", idx_end)
        
    print("last idx_end of eval:", idx_end)
        
    
    if shuffle:
        rng = np.random.RandomState(1234567890)
        shuffle_list = rng.permutation(range(num_image_train))
        dataset_image_train = dataset_image_train[shuffle_list,:,:,:]
        
        shuffle_list = rng.permutation(range(num_image_eval))
        dataset_image_eval = dataset_image_eval[shuffle_list,:,:,:]


def get_batch(idx, train=True):
    
    if (train==True):
        dataset_image = dataset_image_train
    else:
        dataset_image = dataset_image_eval
        
    x_batch = np.array(np.zeros(shape=[batch_size,
                                      time_step_in,
                                      num_sensor,
                                      num_channel]))
    y_batch = np.array(np.zeros(shape=[batch_size,
                                      time_step_out,
                                      num_sensor,
                                      num_channel]))
    
    idx_srt, idx_end = idx*batch_size, (idx+1)*batch_size
    x_batch = dataset_image[idx_srt:idx_end,:time_step_in,:,:]
    y_batch = dataset_image[idx_srt:idx_end,time_step_in:time_step_in+time_step_out,:,:]
         
    y_batch_ft = y_batch.reshape((batch_size,-1))
    
    return x_batch, y_batch_ft
    

def print_activations(t):
  print(t.op.name, ' ', t.get_shape().as_list())


def squash(vector):
    '''Squash function for nonlinearity
    @input
        vector: A tensor with shape [batch_size, :, CAP_caps1_vec_len, 1].
    @output
        A tensor with the same shape as vector but squashed in 'vec_len' dimension.
    '''
    epsilon = 1e-9
    vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keepdims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector  # element-wise
    return(vec_squashed)
    
def routing(input, b_IJ):
    ''' The routing algorithm based on Huadong Liao's implementation
    @input
        input: A Tensor with shape [batch_size, num_capsule = -1, 1, CAP_caps1_vec_len, 1]
        b_IJ: A Tensor with shape [batch_size, num_capsule, CAP_caps2_num_outputs, 1, 1]
            'num_capsule' is the number of capsules in PrimaryCaps layer
            'CAP_caps2_num_outputs' is the number of capsules in TrafficCaps layer
    @output
        A Tensor with shape [batch_size, CAP_caps2_num_outputs, CAP_caps2_vec_len, 1]
    Notes:
        u_i represents the vector output of capsule i in the layer l, and
        v_j the vector output of capsule j in the layer l+1.
     '''
    # W: [1, num_caps_i, num_caps_j * len_v_j, len_u_j, 1]
    CAP_caps2_num_outputs = num_sensor*time_step_out # number of outputs in the second capsule layer
    num_caps_j = CAP_caps2_num_outputs*CAP_caps2_vec_len*num_channel
    num_caps_i = int(CAP_caps1_num_filter*time_step_in*num_sensor/CAP_caps1_vec_len)
    
    W = tf.get_variable('Weight', shape=(1,num_caps_i,num_caps_j,CAP_caps1_vec_len, 1), dtype=tf.float32,
                        initializer=tf.random_normal_initializer(stddev=CAP_routing_stddev))
    biases = tf.get_variable('bias', shape=(1, 1, CAP_caps2_num_outputs, CAP_caps2_vec_len, 1))

    # Since tf.matmul is a time-consuming op,
    # A better solution is using element-wise multiply, reduce_sum and reshape
    # ops instead. Matmul [a, b] x [b, c] is equal to a series ops as
    # element-wise multiply [a*c, b] * [a*c, b], reduce_sum at axis=1 and
    # reshape to [a, c]
    input = tf.tile(input, [1, 1, num_caps_j, 1, 1])

    u_hat = tf.reduce_sum(W * input, axis=3, keepdims=True)
    u_hat = tf.reshape(u_hat, shape=[-1, num_caps_i, CAP_caps2_num_outputs, CAP_caps2_vec_len, 1])

    # In forward, u_hat_stopped = u_hat; in backward, no gradient passed back from u_hat_stopped to u_hat
    u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

    # routing iteration
    for r_iter in range(CAP_iter_routing):
        with tf.variable_scope('iter_' + str(r_iter)):
           
            c_IJ = tf.nn.softmax(b_IJ, axis=2)

            # At last iteration, use `u_hat` in order to receive gradients from the following graph
            if r_iter == CAP_iter_routing - 1:
               
                # weighting u_hat with c_IJ, element-wise in the last two dims
                s_J = tf.multiply(c_IJ, u_hat)
                # then sum in the second dim, resulting in [batch_size, 1, 10, 16, 1]
                s_J = tf.reduce_sum(s_J, axis=1, keepdims=True) + biases

                v_J = squash(s_J)
                v_J = s_J

            elif r_iter < CAP_iter_routing - 1:  # Inner iterations, do not apply backpropagation
                s_J = tf.multiply(c_IJ, u_hat_stopped)
                s_J = tf.reduce_sum(s_J, axis=1, keepdims=True) + biases
                v_J = squash(s_J)
                v_J = s_J
                
                v_J_tiled = tf.tile(v_J, [1, num_caps_i, 1, 1, 1])
                u_produce_v = tf.reduce_sum(u_hat_stopped * v_J_tiled, axis=3, keepdims=True)

                # b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)
                b_IJ += u_produce_v

    return(v_J)


if __name__ == '__main__':
    tf.app.run(main=main)
