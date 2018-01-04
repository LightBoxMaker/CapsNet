import os
import tensorflow as tf
from tqdm import tqdm

from config import cfg
import numpy as np
from utils import load_mnist_affNIST
from capsNet import CapsNet





def main(_):
#     print("PrePreA")
    capsNet = CapsNet(is_training=cfg.is_training)
#     print("PreA")
    tf.logging.info('Graph loaded')
    sv = tf.train.Supervisor(graph=capsNet.graph,
                             logdir=cfg.logdir,
                             save_model_secs=0,
                            summary_op = None)

    path = cfg.results + '/accuracy.csv'
    if not os.path.exists(cfg.results):
        os.mkdir(cfg.results)
    elif os.path.exists(path):
        os.remove(path)

    fd_results = open(path, 'w')
    fd_results.write('step,test_acc\n')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    ex_imgs, ex_imgs_label, imgs_scale,imgs_rotated,imgs_label = load_mnist_affNIST(True)
    teX, teY, teCtrl = load_mnist_affNIST(False)
    num_batches = ex_imgs.shape[0] + imgs_rotated.shape[0]  + imgs_scale.shape[0]
    
    
#     print("A")
    with sv.managed_session(config=config) as sess:
        num_test_batch = teX.shape[0] // cfg.batch_size
            
        for epoch in range(cfg.epoch):
            
            # Recreate/reshuffle batches for each epoch
            var_seq = np.random.choice(range(6),size = num_batches,replace = True) # select 6 to attain 1:1:4 ratio
            imgs_rotated_select = np.random.choice(range(imgs_rotated.shape[0]),size = sum(var_seq == 0), replace = True)
            imgs_scale_select = np.random.choice(range(imgs_scale.shape[0]),size = sum(var_seq == 1), replace = True)
#             imgs_hscale_select = np.random.choice(range(imgs_hscale.shape[0]),size = sum(var_seq == 2), replace = True)
            ex_imgs_select = np.random.choice(range(ex_imgs.shape[0]),size = sum(var_seq >= cfg.num_ex_var), replace = True)
            var_seq = np.array(sorted(var_seq))

            epoch_batches_imgs = np.concatenate([imgs_rotated[imgs_rotated_select],
                   imgs_scale[imgs_scale_select],
                   ex_imgs[ex_imgs_select]],axis = 0)
            epoch_batches_imgs = epoch_batches_imgs.reshape([num_batches,cfg.batch_size,cfg.img_dim,cfg.img_dim,1])

            epoch_batches_labels = np.concatenate([imgs_label[imgs_rotated_select],
                           imgs_label[imgs_scale_select],
                           ex_imgs_label[ex_imgs_select]],axis = 0)

            shuffle_ind = np.random.choice(range(num_batches),size = num_batches,replace = False)
            var_seq = var_seq[shuffle_ind]
            epoch_batches_imgs = epoch_batches_imgs[shuffle_ind]
            epoch_batches_labels = epoch_batches_labels[shuffle_ind]
        
        
            if sv.should_stop():
                break
            for step in tqdm(range(num_batches), total=num_batches, ncols=70, leave=False, unit='b'):

                _, global_step = sess.run([capsNet.train_op,capsNet.global_step],
                                          feed_dict = {capsNet.X: epoch_batches_imgs[step],
                                                        capsNet.labels: epoch_batches_labels[step],
                                                      capsNet.train_var: var_seq[step]})
                

                if step % cfg.train_sum_freq == 0:
                    _, summary_str = sess.run([capsNet.train_op, capsNet.train_summary],
                                             feed_dict = {capsNet.X: epoch_batches_imgs[step],
                                                    capsNet.labels: epoch_batches_labels[step],
                                                          capsNet.train_var: var_seq[step]})
                    sv.summary_writer.add_summary(summary_str, global_step)

                if (global_step + 1) % cfg.test_sum_freq == 0:
                    test_acc = 0
                    for i in range(num_test_batch):
                        start = i * cfg.batch_size
                        end = start + cfg.batch_size
                        test_acc += sess.run(capsNet.batch_accuracy, {capsNet.X: teX[start:end], 
                                                                      capsNet.labels: teY[start:end]})
                    test_acc = test_acc / (cfg.batch_size * num_test_batch)
                    fd_results.write(str(global_step + 1) + ',' + str(test_acc) + '\n')
                    fd_results.flush()

                if global_step % cfg.save_freq == 0:
                    sv.saver.save(sess, cfg.logdir + '/model_epoch_%04d_step_%02d' % (epoch, global_step))
                
                
        sv.saver.save(sess, cfg.logdir + '/Final_model_epoch_%02d_step_%04d' % (epoch, global_step))

    fd_results.close()
    tf.logging.info('Training done')


if __name__ == "__main__":
    tf.app.run()
