import os
import scipy.io
import time
import tensorflow as tf
import numpy as np

from functools import reduce
from operator import mul

def nn_train(nn, config):
    print ("Loading .tfrecords data...", config.dataset)
    if not os.path.exists(config.dataset):
        raise Exception("training data not find.")
        return
          
    time_str = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime(time.time()))    
    train_file_path = os.path.join(config.dataset, "train_sample_batches_*")    
    train_files = tf.train.match_filenames_once(train_file_path)
    filename_queue = tf.train.string_input_producer(train_files, shuffle = True)
    
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, 
                                       features = {'ml': tf.FixedLenFeature([], tf.string),
                                                   'mh': tf.FixedLenFeature([], tf.string),
                                                   'd_bone': tf.FixedLenFeature([], tf.string),
                                                   'd_tissue': tf.FixedLenFeature([], tf.string),
                                                   'rl': tf.FixedLenFeature([], tf.string),
                                                   'rh': tf.FixedLenFeature([], tf.string)})
    
    ml = tf.decode_raw(features['ml'], tf.float32)
    mh = tf.decode_raw(features['mh'], tf.float32)
    d1 = tf.decode_raw(features['d_bone'], tf.float32)
    d2 = tf.decode_raw(features['d_tissue'], tf.float32)
    rl = tf.decode_raw(features['rl'], tf.float32)
    rh = tf.decode_raw(features['rh'], tf.float32)
   
    ml = tf.reshape(ml, [nn.input_height, nn.input_width, nn.num_channel])
    mh = tf.reshape(mh, [nn.input_height, nn.input_width, nn.num_channel])

    d1 = tf.reshape(d1, [512, 512, nn.output_depth])
    d2 = tf.reshape(d2, [512, 512, nn.output_depth])
    rl = tf.reshape(rl, [512, 512, nn.output_depth])
    rh = tf.reshape(rh, [512, 512, nn.output_depth])
    com_resize = 256
    d1_resized = tf.image.resize_images(d1, [com_resize, com_resize], method = 0)
    d2_resized = tf.image.resize_images(d2, [com_resize, com_resize], method = 0)
    rl_resized = tf.image.resize_images(rl, [com_resize, com_resize], method = 0)
    rh_resized = tf.image.resize_images(rh, [com_resize, com_resize], method = 0)
    
    
    ml_batch, mh_batch, d1_batch, d2_batch, rl_batch, rh_batch = tf.train.shuffle_batch([ml, mh, d1_resized, d2_resized, rl_resized, rh_resized], 
                                                    batch_size = config.batch_size, 
                                                    capacity = config.batch_size*3 + 50,
                                                    min_after_dequeue = 30)
    
    # loss function define
    d1_batch_pred, d2_batch_pred, rl_batch_pred, rh_batch_pred = nn.feedforward(ml_batch, mh_batch, rl_batch, rh_batch)
    
    global_step = tf.Variable(0, trainable = False)
    variable_averages = tf.train.ExponentialMovingAverage(config.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    
    mse_loss = tf.reduce_mean(0.005*tf.square(d1_batch_pred - d1_batch) / 2 + 0.005*tf.square(d2_batch_pred - d2_batch) / 2 + 
                              tf.square(rl_batch_pred - rl_batch) / 2 + tf.square(rh_batch_pred - rh_batch) / 2)
                            
    tf.add_to_collection('losses', mse_loss)
    tf.summary.scalar('MSE_losses', mse_loss)
    
    learning_rate = tf.train.exponential_decay(config.lr, global_step, 
                                               config.sampleNum / config.batch_size, config.learning_rate_decay)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(mse_loss, global_step = global_step)

    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name = 'train')            

    merged = tf.summary.merge_all()
    summary_path = os.path.join(config.summary_dir, time_str)
    os.mkdir(summary_path)
    
    run_config = tf.ConfigProto(allow_soft_placement = True)
    run_config.gpu_options.allow_growth = True
    
    num_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    print("Number of trainable parameters: %d"%(num_params))
    
    with tf.Session(config = run_config) as sess:    
        summary_writer = tf.summary.FileWriter(summary_path, sess.graph)    
        saver = tf.train.Saver()
        if config.goon:
            if not os.path.exists(config.checkpoint):
                raise Exception("checkpoint path not find.")
                return
            print("Loading trained model... ", config.checkpoint)
            ckpt = tf.train.get_checkpoint_state(config.checkpoint)
            saver.retore(sess, ckpt)
        else:
            init = (tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init)
            
        save_model_path = os.path.join(config.output_model_dir, time_str)
        os.mkdir(save_model_path)
    
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess = sess, coord = coord)
        
        variable_name = [v.name for v in tf.trainable_variables()]
        print(variable_name)
        
        print("start training sess...")
        start_time = time.time()
        for i in range(config.iteration):
            batch_start_time = time.time()
            summary, _, loss_value, step, learningRate = sess.run([merged, train_op, mse_loss, global_step, learning_rate])
            batch_end_time = time.time()
            sec_per_batch = batch_end_time - batch_start_time
            if step % config.summary_step == 0:
                summary_writer.add_summary(summary, step//config.summary_step)
            if step % config.model_step == 0:
                print("Saving model (after %d iteration)... " %(step))
                saver.save(sess, os.path.join(save_model_path, config.model_name + ".ckpt"), global_step = global_step)
            print("sec/batch(%d) %gs, global step %d batches, training epoch %d/%d, learningRate %g, loss on training is %g" 
                  % (config.batch_size, sec_per_batch, step, i*config.batch_size / config.sampleNum,
                     config.epoch, learningRate, loss_value))
            print("Elapsed time: %gs" %(batch_end_time - start_time))
            
        coord.request_stop()
        coord.join(threads)
        summary_writer.close()
        print("Train done. ")
        print("Saving model... ", save_model_path)
        saver.save(sess, os.path.join(save_model_path, config.model_name + ".ckpt"), global_step = global_step)
        sess.close()
        
    print("Sess closed.")

def nn_test(nn, config):
    print ("Loading .tfrecords data...", config.dataset)
    if not os.path.exists(config.dataset):
        raise Exception(".mat file not find.")
        return

def nn_feedforward(nn, config):
    print ("Loading .mat data...", config.dataset)
    if not os.path.exists(config.dataset):
        print("Testing .mat file not find.")
        raise Exception("Testing .mat file not find.")
        return
    
    run_config = tf.ConfigProto()
    with tf.Session(config = run_config) as sess:
        mat_file_list = []
        filelist = os.listdir(config.dataset)
        for line in filelist:
            file = os.path.join(config.dataset, line)
            if os.path.isfile(file):
                mat_file_list.append(file)
                print(file)
        
        ml = tf.placeholder(tf.float32, [1, 360, 1024, 1], name = 'xl-input')
        mh = tf.placeholder(tf.float32, [1, 360, 1024, 1], name = 'xh-input')
        
        d1_pred, d2_pred, rl_pred, rh_pred = nn.feedforward(ml, mh)
        
        variable_averages = tf.train.ExponentialMovingAverage(config.moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        
        ckpt = tf.train.get_checkpoint_state(config.checkpoint)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-'[-1])
            print('global step: ', global_step)            
            
            file_num = 1
            for mat_file in mat_file_list:
                print('loading file: ', mat_file)
                mat = scipy.io.loadmat(mat_file)                
                mat_ml = mat['ml']
                mat_mh = mat['mh']

                dim_h = mat_ml.shape[0]
                dim_w = mat_ml.shape[1]
                if mat_ml.ndim < 3:
                    mat_ml = mat_ml.reshape(dim_h, dim_w, 1)
                    mat_mh = mat_mh.reshape(dim_h, dim_w, 1)                    
                img_num = mat_ml.shape[2]
                print('image num: ', img_num)
                
                d1 = np.zeros((nn.output_h, nn.output_w, img_num), dtype = np.float32)
                d2 = np.zeros((nn.output_h, nn.output_w, img_num), dtype = np.float32)
                rl = np.zeros((nn.output_h, nn.output_w, img_num), dtype = np.float32)
                rh = np.zeros((nn.output_h, nn.output_w, img_num), dtype = np.float32)
         
                input_ml = np.zeros( (1, dim_h, dim_w, 1) )
                input_mh = np.zeros( (1, dim_h, dim_w, 1) )
            
                start_time = time.time()
                for i in range( int(img_num) ):
                    input_ml = mat_ml[:,:,i].reshape([1, dim_h, dim_w, 1])
                    input_mh = mat_mh[:,:,i].reshape([1, dim_h, dim_w, 1])
                    
                    output_d1, output_d2, output_rl, output_rh = sess.run([d1_pred, d2_pred, rl_pred, rh_pred], \
                                                                          feed_dict={ml: input_ml, mh: input_mh})
                    output_d1.resize(1, nn.output_h, nn.output_w, 1)
                    output_d2.resize(1, nn.output_h, nn.output_w, 1)
                    output_rl.resize(1, nn.output_h, nn.output_w, 1)
                    output_rh.resize(1, nn.output_h, nn.output_w, 1)                  
                    
                    d1[:,:,i] = output_d1.reshape([nn.output_h, nn.output_w])
                    d2[:,:,i] = output_d2.reshape([nn.output_h, nn.output_w])
                    rl[:,:,i] = output_rl.reshape([nn.output_h, nn.output_w])
                    rh[:,:,i] = output_rh.reshape([nn.output_h, nn.output_w])
                    
                end_time = time.time()
                print("Elapsed time: %gs" %(end_time - start_time))
    
                result_fileName = config.output_data_dir + '/' + 'result_test_%.4d.mat' %(file_num)
                scipy.io.savemat(result_fileName, {'d1':d1, 'd2':d2, 'rl':rl, 'rh':rh})
                print('Feed forward done. Result: ', result_fileName)
                file_num = file_num + 1
        else:
            print('No checkpoint file found.')
            return
