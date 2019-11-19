
import tensorflow as tf
import scipy.io as sio
import numpy as np
import os
import scipy.sparse as sparse
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


block_size = 48
channel = 31


batch_size = 25
PhaseNumber = 9

EpochNum = 160


print('Load Data...')

import h5py

Training_data_Name = 'Training_Data.mat'
Training_data = h5py.File(Training_data_Name)
Training_labels = Training_data['label']
Training_labels = np.transpose(Training_labels, (0, 3, 2, 1))
del Training_data
nrtrain = Training_labels.shape[0]
print(nrtrain)
print(Training_labels.shape)
gloabl_steps = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(learning_rate=0.001, global_step=gloabl_steps, decay_steps=(nrtrain//batch_size)*10, decay_rate=0.9,
                                           staircase=True)



Cu = tf.placeholder(tf.float32, [None, block_size, block_size, channel])
X_output = tf.placeholder(tf.float32, [None, block_size, block_size, channel])
b = tf.zeros(shape=(tf.shape(X_output)[0], channel-1, tf.shape(X_output)[2], tf.shape(X_output)[3]))


def add_con2d_weight_bias(w_shape, b_shape, order_no):
    Weights = tf.get_variable(shape=w_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(), name='Weights_%d' % order_no)
    biases = tf.Variable(tf.random_normal(b_shape, stddev=0.05), name='biases_%d' % order_no)
    return [Weights, biases]


def Encode_procedure(x):
    y = tf.multiply(x, Cu)
    y = tf.reduce_sum(y, axis=3)
    return y




def Recon_block(xt, x0, layer_no, corr=None):

    deta = tf.Variable(0.04, dtype=tf.float32, name='deta_%d' % layer_no)
    eta = tf.Variable(0.8, dtype=tf.float32, name='eta_%d' % layer_no)

    wz1 = tf.Variable(0.8, dtype=tf.float32, name='wz1_%d' % layer_no)
    channelNum = channel

    # local_module
    filter_size1 = 3
    filter_num = 64

    [Weights_0, bias_0] = add_con2d_weight_bias([filter_size1, filter_size1, channelNum, filter_num], [filter_num], 0)
    [Weights_1, bias_1] = add_con2d_weight_bias([filter_size1, filter_size1, filter_num, channelNum], [channelNum], 1)

    x_resx1 = tf.nn.relu(tf.nn.conv2d(xt, Weights_0, strides=[1, 1, 1, 1], padding='SAME'))
    x_resx2 = tf.nn.conv2d(x_resx1, Weights_1, strides=[1, 1, 1, 1], padding='SAME')
    z1 = xt + x_resx2

    # Non -Local Module
    filter_size2 = 1
    [Weights_g, bias_g] = add_con2d_weight_bias([filter_size2, filter_size2, channelNum, channelNum], [channelNum], 3)  # G
    x_g = tf.nn.conv2d(xt, Weights_g, strides=[1, 1, 1, 1], padding='SAME')

    x_theta_reshaped = tf.reshape(xt, [tf.shape(xt)[0], tf.shape(xt)[1] * tf.shape(xt)[2],
                                       tf.shape(xt)[3]])
    x_phi_reshaped = tf.reshape(xt, [tf.shape(xt)[0], tf.shape(xt)[1] * tf.shape(xt)[2], tf.shape(xt)[3]])
    x_phi_permuted = tf.transpose(x_phi_reshaped, perm=[0, 2, 1])
    x_mul1 = tf.matmul(x_theta_reshaped, x_phi_permuted)
    if corr is not None:
        x_mul1 += corr

    x_mul1_softmax = tf.scalar_mul(1/(block_size*block_size), x_mul1)
    x_g_reshaped = tf.reshape(x_g, [tf.shape(x_g)[0], tf.shape(x_g)[1] * tf.shape(x_g)[2], tf.shape(x_g)[3]])
    x_mul2 = tf.matmul(x_mul1_softmax, x_g_reshaped)
    z2 = tf.reshape(x_mul2, [tf.shape(xt)[0], tf.shape(xt)[1], tf.shape(xt)[2], tf.shape(xt)[3]])

  

    z = tf.scalar_mul(wz1, z1) + tf.scalar_mul((1-wz1), z2)

    yt = tf.multiply(xt, Cu)
    yt = tf.reduce_sum(yt, axis=3)
    yt1 = tf.expand_dims(yt, axis=3)
    yt2 = tf.tile(yt1, [1, 1, 1, channel])
    xt2 = tf.multiply(yt2, Cu)  # PhiT*Phi*xt
    x = tf.scalar_mul(1-deta*eta, xt) - tf.scalar_mul(deta, xt2) + tf.scalar_mul(deta, x0) + tf.scalar_mul(deta*eta, z)
    return x, x_mul1




def inference_ista( x, n, reuse):
    xt = x
    for i in range(n):
        with tf.variable_scope('Phase_%d' %i, reuse=reuse):
            if i == 0:
                xt, corr = Recon_block(xt, x, i)
            else:
                xt, corr = Recon_block(xt, x, i, corr)
    return xt



y = Encode_procedure(X_output)
y1 = tf.expand_dims(y, axis=3)
y2 = tf.tile(y1, [1, 1, 1, channel])
x0 = tf.multiply(y2, Cu)

Prediction = inference_ista(x0, PhaseNumber, reuse=False)

cost_all = tf.reduce_mean(tf.square(Prediction - X_output))

optm_all = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_all, global_step=gloabl_steps)

init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

sess = tf.Session(config=config)
sess.run(init)


print("...............................")
print("Phase Number is %d, block_size is %d%%" % (PhaseNumber, block_size))
print("...............................\n")

print("Strart Training..")


model_dir = 'HSIRecon_DeepNonlocalUnrolling_ICVL_%dPhase' % (PhaseNumber)


output_file_name = "Log_output_%s.txt" % (model_dir)

for epoch_i in range(EpochNum):
    randidx_all = np.random.permutation(nrtrain)
    for batch_i in range(nrtrain // batch_size):
        randidx = randidx_all[batch_i*batch_size:(batch_i+1)*batch_size]
        batch_ys = Training_labels[randidx, :, :, :]
	Cu_input = np.zeros([block_size, block_size, channel])
	T = np.round(np.random.rand(block_size/2, block_size/2))
	T = np.concatenate([T,T],axis=0)
	T = np.concatenate([T,T],axis=1)
	for ch in range(channel):
        	Cu_input[:,:,ch] = np.roll(T, shift=ch, axis=0)
	Cu_input = np.expand_dims(Cu_input, axis=0)
	Cu_input = np.tile(Cu_input, [batch_size, 1, 1, 1])

	feed_dict = {X_output: batch_ys, Cu: Cu_input}
    sess.run(optm_all, feed_dict=feed_dict)
    	

    output_data = "[%02d/%02d/%02d] cost: %.4f  learningrate: %.6f \n" % (batch_i, nrtrain // batch_size, epoch_i, sess.run(cost_all, feed_dict=feed_dict), sess.run(learning_rate, feed_dict=feed_dict))
    print(output_data)

    output_file = open(output_file_name, 'a')
    output_file.write(output_data)
    output_file.close()

    writer = tf.summary.FileWriter('logs_Correlation_Random_mask/', tf.get_default_graph())
    writer.close()

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    saver.save(sess, './%s/CS_Saved_Model_%d.cpkt' % (model_dir, epoch_i), write_meta_graph=False)

print("Training Finished")
sess.close()