import os
import math

def config_par(config):
    config.DEFINE_string("output_model_dir", "../model", "")
    config.DEFINE_string("output_data_dir", "../result", "")
    config.DEFINE_string("model_name", "lstm", "")
    config.DEFINE_integer("sampleNum", 3226, "")
    config.DEFINE_integer("iteration", math.ceil( config.FLAGS.sampleNum / config.FLAGS.batch_size * config.FLAGS.epoch ), "")
    
    config.DEFINE_string("summary_dir", "../log", "")
    config.DEFINE_integer("summary_step", 20, "")
    
    config.DEFINE_float("learning_rate_decay", 0.85, "")
    config.DEFINE_float("moving_average_decay", 0.99, "")
    config.DEFINE_float("regularazition_rate", 0.0001, "")
    
    if not os.path.exists(config.FLAGS.output_model_dir):
        os.makedirs(config.FLAGS.output_model_dir)
    if not os.path.exists(config.FLAGS.output_data_dir):
        os.makedirs(config.FLAGS.output_data_dir)
    if not os.path.exists(config.FLAGS.summary_dir):
        os.makedirs(config.FLAGS.summary_dir)
        
    return config