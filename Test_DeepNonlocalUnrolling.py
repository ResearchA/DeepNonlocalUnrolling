
import tensorflow as tf
import scipy.io as sio
import numpy as np
import glob
from time import time
from PIL import Image
import math
import h5py
import scipy.sparse as sparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"       

cpkt_model_number = 112
noiseSigma = 0.0
PhaseNumber = 9

model_dir = 'HSIRecon_DeepNonlocalUnrolling_ICVL_%dPhase' % (PhaseNumber)



out_filePath = './Result/ICVL/'

block_size = 48
channel = 31


batch_size = 30
learning_rate = 0.0001
EpochNum = 2000


import h5py
Cu_data_Name = 'Cu_%d.mat' % block_size  
Cu_data = sio.loadmat(Cu_data_Name)
Cu_input = Cu_data['Cu']
print(Cu_input.shape)


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
    y = y + noiseSigma * tf.random_normal([tf.shape(y)[0], tf.shape(y)[1], tf.shape(y)[2]])
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

optm_all = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_all)

init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

sess = tf.Session(config=config)
# sess.run(init)

saver.restore(sess, './%s/CS_Saved_Model_%d.cpkt' % (model_dir, cpkt_model_number))

Test_Img = './HSITest/Harvard48'
filepaths = os.listdir(Test_Img)
ImgNum = len(filepaths)

batch = 77
index = 0
for trainable_variable in tf.trainable_variables():
    temp = sess.run(trainable_variable)
    if(temp.shape == ()):
        print(temp)
        index+=1
    if(index==3):
        print('==================')
        index = 0
Cu_input = np.zeros([block_size, block_size, channel])
T = np.round(np.random.rand(block_size/2, block_size/2))
T = np.concatenate([T,T],axis=0)
T = np.concatenate([T,T],axis=1)
for ch in range(channel):
        Cu_input[:,:,ch] = np.roll(T, shift=-ch, axis=0)
Cu_input = np.expand_dims(Cu_input, axis=0)
Cu_input = np.tile(Cu_input, [batch, 1, 1, 1])

for img_no in range(ImgNum ):

    imgName = filepaths[img_no]
    imgName = imgName[0:-4]
    testData = sio.loadmat(Test_Img+'/'+filepaths[img_no])
    Hyper_image = testData['hyper_image']
    patch_image = testData['patch_image']
    patch_image = np.transpose(patch_image, (3, 0, 1, 2))

    patchNum = patch_image.shape[0]

    for i in range(patchNum // batch):
        start = time()
		xoutput = patch_image[i * batch:(i + 1) * batch]

        Prediction_value = sess.run(Prediction, feed_dict={X_output: xoutput, Cu: Cu_input})
        end = time()
        y_value = sess.run(y, feed_dict={X_output: xoutput, Cu: Cu_input})
        cost_all_value = sess.run(cost_all, feed_dict={X_output: xoutput, Cu: Cu_input})
        print("Run time for %s is %.4f, loss sym is %.4f" % (imgName, (end - start), cost_all_value))
        Prediction_patch = np.transpose(Prediction_value, (1, 2, 3, 0))
        if i == 0:
            output = Prediction_patch
            y_out = y_value
        else:
            output = np.concatenate([output, Prediction_patch], axis=3)
            y_out = np.concatenate([y_out, y_value], axis=0)
        print(output.shape)

    out_dict = {'output': output,
                'y_out':y_out,
                'label': Hyper_image}
    if not os.path.exists(out_filePath):
        os.makedirs(out_filePath)
    out_filename = out_filePath + np.str(imgName)+'.mat'
    sio.savemat(out_filename, out_dict)

sess.close()

print("Reconstruction READY")