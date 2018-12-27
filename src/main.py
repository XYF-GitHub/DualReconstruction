import os
import sys
import tensorflow as tf
import pprint

from model import DualReconstructionModel
from solver import nn_train, nn_test, nn_feedforward
from config import config_par

flags = tf.app.flags
flags.DEFINE_string("dataset", "data", ".tfRecord training file or .mat testing file [data]")
flags.DEFINE_string("mode", "train", "train, test or feedforward [train]")
flags.DEFINE_integer("epoch", 30, "Epoch to train [500]")
flags.DEFINE_integer("batch_size", 10, "The size of batch images [10]")
flags.DEFINE_integer("model_step", 1000, "The number of iteration to save model [1000]")
flags.DEFINE_float("lr", 0.01, "The base learning rate [0.01]")
flags.DEFINE_string("checkpoint", "checkpoint", "The path of checkpoint")
flags.DEFINE_boolean("goon", False, "Go on training flag [0]")

def main(_):
    os.chdir(sys.path[0])
    print("Current cwd: ", os.getcwd())
    
    config = config_par(flags)    
    pp = pprint.PrettyPrinter()    
    pp.pprint(config.FLAGS.__flags)    
    FLAGS = config.FLAGS
    
    print("Building model ...")
    nn = DualReconstructionModel()    
    if FLAGS.mode == "train":
        print("Mode: train ...")
        nn_train(nn, FLAGS)
    elif FLAGS.mode == "test":
        print("Mode: test ...")
        nn_test(nn, FLAGS)
    elif FLAGS.mode == "feedforward":
        print("Mode: feedforward ...")
        nn_feedforward(nn, FLAGS)
            
    exit()

if __name__ == '__main__':
    tf.app.run()