import tensorflow as tf

import numpy as np
import odl
import odl.contrib.tensorflow

from cell import ConvLSTMCell

import layer_xyf

# input size
prjLen = 1024
view_num = 360
channel = 1

# output size
output_h = 256
output_w = 256
output_depth = 1

# LSTM hidden size
filters_lstm = 64
kernal_lstm = [3, 3]

#conv: [filter_height, filter_width, in_channels, out_channels]
filter_shape = [256, 256, filters_lstm, 1]

sod = 1000
sdd = 1500

iteration = 10

# CNN
#layer_1 input_size 128*128 / 256*256 / 512*512   conv: [filter_height, filter_width, in_channels, out_channels]
FILTER_1 = [3, 3, 1, 32]
STRIDE_1 = 2
PAD_1 = "SAME"

#layer_2 input_size 64*64 / 128*128 / 256*256
FILTER_2 = [3, 3, 32, 32]
STRIDE_2 = 2
PAD_2 = "SAME"

#layer_3 input_size 32*32 / 64*64 / 128*128
FILTER_3 = [3, 3, 32, 32]
STRIDE_3 = 2
PAD_3 = "SAME"

#layer_4 input_size 16*16 / 32*32 / 64*64
#FILTER_4 = [9, 9, 256, 512]
FILTER_4 = [3, 3, 32, 32]
STRIDE_4 = 2
PAD_4 = "SAME"

#layer_5 input_size 8*8 / 16*16 / 32*32
FILTER_5 = [16, 16, 32, 32]
STRIDE_5 = 1
PAD_5 = "VALID"

#layer_6 input_size 1*1*128
FILTER_6 = [1, 1, 64, 1]
STRIDE_6 = 1
PAD_6 = "VALID"

space = odl.uniform_discr([-256, -256], [256, 256], [output_h, output_w], dtype='float32')
angle_partition = odl.uniform_partition(0, 2 * np.pi, view_num)
detector_partition = odl.uniform_partition(-360, 360, prjLen)
geometry = odl.tomo.FanFlatGeometry(angle_partition, detector_partition, 
                                            src_radius = sod, det_radius = sdd - sod)
operator = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

odl_op_layer = odl.contrib.tensorflow.as_tensorflow_layer(operator, 'RayTransform')
odl_op_layer_adjoint = odl.contrib.tensorflow.as_tensorflow_layer(operator.adjoint, 'RayTransformAdjoint')


class DualReconstructionModel(object):
    def __init__(self):
        self.input_width = prjLen
        self.input_height = view_num
        self.num_channel = channel
        self.output_h = output_h
        self.output_w = output_w
        self.output_depth = output_depth
        
    def feedforward(self, ml, mh, regularizer = 0):
        input_shape = mh.get_shape().as_list()
        batch_size = input_shape[0]
          
        #LSTM
        with tf.variable_scope('LSTM_1'):
            rh = tf.get_variable("rh", [batch_size, self.output_h, self.output_w, self.output_depth], initializer = tf.constant_initializer(0.0), trainable = False)          
            conv_w1 = tf.get_variable( "weight_lstm1", filter_shape, initializer = tf.truncated_normal_initializer(stddev = 0.001) )
            conv_b1 = tf.get_variable( "bias1", 1, initializer = tf.constant_initializer(0.0) )              
            cell_g1 = ConvLSTMCell([self.output_h, self.output_w], filters_lstm, kernal_lstm)
            init_state_g1 = cell_g1.zero_state(batch_size, dtype = tf.float32)
            state1 = init_state_g1            
            for timestep in range(iteration):
                if timestep > 0:
                    tf.get_variable_scope().reuse_variables()
                rh_tr = tf.transpose(rh, perm = [0, 2, 1, 3])
                g1 = odl_op_layer_adjoint( (odl_op_layer(rh_tr) - mh) )
                gt1 = tf.transpose(g1, perm = [0, 2, 1, 3])
                (cell_output1, state1) = cell_g1(gt1, state1)
                conv1 = tf.nn.conv2d(cell_output1, conv_w1, [1, 1, 1, 1], "VALID")
                s1 = tf.nn.tanh(tf.nn.bias_add(conv1, conv_b1))
                self.variable_summaries(s1, ('s1_%d'%timestep))
               
                rh = rh + 0.0001*s1*gt1
                rh = tf.clip_by_value(rh, 0, 5)
                tf.summary.image('rh_pred_%d'%timestep, rh, 1)
                    
        with tf.variable_scope('LSTM_2'):
            rl = tf.get_variable("rl", [batch_size, self.output_h, self.output_w, self.output_depth], initializer = tf.constant_initializer(0.0), trainable = False)       
            conv_w2 = tf.get_variable("weight_lstm2", filter_shape, initializer = tf.truncated_normal_initializer(stddev = 0.001) )
            conv_b2 = tf.get_variable("bias2", 1, initializer = tf.constant_initializer(0.0) )               
            cell_g2 = ConvLSTMCell([self.output_h, self.output_w], filters_lstm, kernal_lstm)
            init_state_g2 = cell_g2.zero_state(batch_size, dtype = tf.float32)
            state2 = init_state_g2          
            for timestep in range(iteration):
                if timestep > 0:
                    tf.get_variable_scope().reuse_variables()
                rl_tr = tf.transpose(rl, perm = [0, 2, 1, 3])
                g2 = odl_op_layer_adjoint( (odl_op_layer(rl_tr) - ml) )
                gt2 = tf.transpose(g2, perm = [0, 2, 1, 3])
                
                (cell_output2, state2) = cell_g2(gt2, state2)
                conv2 = tf.nn.conv2d(cell_output2, conv_w2, [1, 1, 1, 1], "VALID")
                s2 = tf.nn.tanh(tf.nn.bias_add(conv2, conv_b2))
                self.variable_summaries(s2, ('s2_%d'%timestep))
                    
                rl = rl + 0.0001*s2*gt2
                rl = tf.clip_by_value(rl, 0, 5)
                tf.summary.image('rl_pred_%d'%timestep, rl, 1)
                
         #CNN
        layer_L1 = layer_xyf.convo(rl, "conv_L1", FILTER_1, STRIDE_1, PAD_1)
        layer_L2 = layer_xyf.convo(layer_L1, "conv_L2", FILTER_2, STRIDE_2, PAD_2)
        layer_L3 = layer_xyf.convo(layer_L2, "conv_L3", FILTER_3, STRIDE_3, PAD_3)
        layer_L4 = layer_xyf.convo(layer_L3, "conv_L4", FILTER_4, STRIDE_4, PAD_4)
        layer_L5 = layer_xyf.convo(layer_L4, "conv_L5", FILTER_5, STRIDE_5, PAD_5)
        
        layer_H1 = layer_xyf.convo(rh, "conv_H1", FILTER_1, STRIDE_1, PAD_1)
        layer_H2 = layer_xyf.convo(layer_H1, "conv_H2", FILTER_2, STRIDE_2, PAD_2)
        layer_H3 = layer_xyf.convo(layer_H2, "conv_H3", FILTER_3, STRIDE_3, PAD_3)
        layer_H4 = layer_xyf.convo(layer_H3, "conv_H4", FILTER_4, STRIDE_4, PAD_4)
        layer_H5 = layer_xyf.convo(layer_H4, "conv_H5", FILTER_5, STRIDE_5, PAD_5)
        
        combine_LH = tf.concat([layer_L5, layer_H5], 3)
    
        pa_pred = layer_xyf.convo_noneRelu(combine_LH, "conv_pa", FILTER_6, STRIDE_6, PAD_6)
        pb_pred = layer_xyf.convo_noneRelu(combine_LH, "conv_pb", FILTER_6, STRIDE_6, PAD_6)
        pc_pred = layer_xyf.convo_noneRelu(combine_LH, "conv_pc", FILTER_6, STRIDE_6, PAD_6)
        pd_pred = layer_xyf.convo_noneRelu(combine_LH, "conv_pd", FILTER_6, STRIDE_6, PAD_6)      
                
        d1 = pa_pred*rh + pb_pred*rl
        d2 = pc_pred*rh + pd_pred*rl
        
        d1 = tf.clip_by_value(d1, 0, 5)
        d2 = tf.clip_by_value(d2, 0, 5)
        
        tf.summary.image('d1_pred', d1, 1)
        tf.summary.image('d2_pred', d2, 1)
        
        return d1, d2, rl, rh

    def variable_summaries(self, var, var_name):
        with tf.name_scope(var_name):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)
        