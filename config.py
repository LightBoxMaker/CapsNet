import tensorflow as tf

flags = tf.app.flags


############################
#    hyper parameters      #
############################

# For separate margin loss
flags.DEFINE_float('m_plus', 0.9, 'the parameter of m plus')
flags.DEFINE_float('m_minus', 0.1, 'the parameter of m minus')
flags.DEFINE_float('lambda_val', 0.5, 'down weight of the loss for absent digit classes')

# for training
flags.DEFINE_integer('batch_size', 20, 'batch size')
flags.DEFINE_integer('epoch', 6, 'epoch')
flags.DEFINE_integer('iter_routing', 3, 'number of iterations in routing algorithm')
flags.DEFINE_boolean('mask_with_y', True, 'use the true label to mask out target capsule or not')

flags.DEFINE_float('stddev', 0.01, 'stddev for W initializer')
flags.DEFINE_float('regularization_scale', 0.005, 'regularization coefficient for reconstruction loss, default to 0.005')
flags.DEFINE_integer('num_ex_var', 2, 'number of extrinsic variable')
flags.DEFINE_float('reg_nudge_to_mean', 0, 'learning rate the gradient for non-train variable converging to mean during a step')


############################
#   environment setting    #
############################
flags.DEFINE_string('dataset', 'data/mnist', 'the path for dataset')
flags.DEFINE_boolean('is_training', True, 'train or predict phase')
flags.DEFINE_integer('num_threads', 8, 'number of threads of enqueueing exampls')
flags.DEFINE_string('logdir', 'logdir', 'logs directory')
flags.DEFINE_integer('train_sum_freq', 5000, 'the frequency of saving train summary(step)')
flags.DEFINE_integer('test_sum_freq', 500, 'the frequency of saving test summary(step)')
flags.DEFINE_integer('save_freq', 3000, 'the frequency of saving model(epoch)')
flags.DEFINE_string('results', 'results', 'path for saving results')
flags.DEFINE_integer('img_dim', 40, 'dimension of the input image')


############################
#   distributed setting    #
############################
flags.DEFINE_integer('num_gpu', 2, 'number of gpus for distributed training')
flags.DEFINE_integer('batch_size_per_gpu', 128, 'batch size on 1 gpu')
flags.DEFINE_integer('thread_per_gpu', 4, 'Number of preprocessing threads per tower.')

cfg = tf.app.flags.FLAGS
# tf.logging.set_verbosity(tf.logging.INFO)
