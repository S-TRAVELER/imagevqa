import tensorflow as tf
import numpy
import time
import math
import logging
import argparse
import os

from data_provision_att_vqa import *
from data_processing_vqa import *
from san_lstm_att_tf import *

options={
    'data_path': '/home/code/imageqa/data_vqa/',
    'feature_file':'trainval_feat.h5',
    'expt_folder': 'expt_1',
    'model_name': 'imageqa',
    'train_split':'trainval1',
    'val_split': 'val2',

    'num_region': 196,
    'region_dim': 512,

    'n_emb':500,
    'n_dim':1024,
    'n_image_feat':512,
    'n_common_feat':500,
    'n_attention':512,
    'n_words':13746,
    'n_output': 1000,

    'max_epochs' : 50,
    'batch_size':100,
    'step_start':100,

    # initialization
    'init_type':'uniform',
    'range': 0.01,
    'std': 0.01,
    # misc
    'gpu_id' : 0,

    'weight_decay': 0.0005,
    'decay_rate': numpy.float32(0.999),
    'drop_ratio' :numpy.float32(0.5),
    'smooth':numpy.float32(1e-8),
    'grad_clip':numpy.float32(0.1),
    'learning_rate': 0.0003,

    'disp_interval': 10,
    'eval_interval': 1000,
    'save_interval': 500  ,

    'sample_answer':True
}

def get_lr(options, curr_epoch):
    if options['optimization'] == 'sqd':
        power = max((curr_epoch - options['step_start'])/options['step'], 0)
        power = math.ceil(power)
        return options['lr']*(options['gamma']**power)
    else:
        return options['lr']

def train(options):
    logger = logging.getLogger('root')
    logger.info(options)
    logger.info('start training')

    data_provision_att_vqa = DataProvisionAttVqa(options['data_path'],
                                                 options['feature_file'])
    
    batch_size = options['batch_size']
    max_epochs = options['max_epochs']

    ###############
    # build model #
    ###############
    
    model = Answer_Generator(options)
    
    tf_loss, tf_aucc, tf_image, tf_question, tf_label = model.build_model()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5, allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    with tf.device('/cpu:0'):
        saver = tf.train.Saver(max_to_keep=100)

    learning_rate=options['learning_rate']
    decay_factor =options['decay_rate']
    tvars = tf.trainable_variables()
    lr = tf.Variable(learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate=lr)
    # gradient clipping
    gvs = opt.compute_gradients(tf_loss,tvars)
    with tf.device('/cpu:0'):
        clipped_gvs = [(tf.clip_by_value(grad, -10.0, 10.0), var) for grad, var in gvs if grad is not None]
    train_op = opt.apply_gradients(clipped_gvs)

    # calculate how many iterations we need
    num_iters_one_epoch = data_provision_att_vqa.get_size(options['train_split']) / batch_size
    max_iters = max_epochs * num_iters_one_epoch
    eval_interval_in_iters = options['eval_interval']
    save_interval_in_iters = options['save_interval']
    disp_interval = options['disp_interval']

    sess.run(tf.initialize_all_variables())

    logging.info('start training...')

    for itr in xrange(max_iters + 1):
        if (itr % eval_interval_in_iters) == 0 or (itr == max_iters):
            timeRecord = -time.time()
            val_loss_list = []
            val_accu_list = []
            val_count = 0
            for batch_image_feat, batch_question, batch_answer_label \
                in data_provision_att_vqa.iterate_batch(options['val_split'],
                                                    batch_size):
                input_idx, input_mask \
                    = process_batch(batch_question,
                                    reverse=options['reverse'])
                batch_image_feat = reshape_image_feat(batch_image_feat,
                                                      options['num_region'],
                                                      options['region_dim'])
                # do the testing process!!!
                loss, accu = sess.run(
                        [tf.loss, tf_aucc],
                        feed_dict={
                            tf_image: batch_image_feat,
                            tf_question: input_idx,
                            tf_label: batch_answer_label.astype('int32').flatten()
                            })
                val_count += batch_image_feat.shape[0]
                val_loss_list.append(loss * batch_image_feat.shape[0])
                val_accu_list.append(accu * batch_image_feat.shape[0])

            ave_val_loss = sum(val_loss_list) / float(val_count)
            ave_val_accu = sum(val_accu_list) / float(val_count)
            timeRecord += time.time()
            logging.info("Iteration: ", itr, " Loss: ", ave_val_loss, " Aucc: ", ave_val_accu ," Learning Rate: ", lr.eval(session=sess))
            logging.info ("Time Cost:", round(timeRecord,2), "s")
    
            # logging.info("Iteration ", itr, " is done. Saving the model ...")
            # saver.save(sess, os.path.join(checkpoint_path, 'model'), global_step=itr)

        if options['sample_answer']:
            batch_image_feat, batch_question, batch_answer_label \
                = data_provision_att_vqa.next_batch_sample(options['train_split'],
                                                       batch_size)
        else:
            batch_image_feat, batch_question, batch_answer_label \
                = data_provision_att_vqa.next_batch(options['train_split'], batch_size)
        input_idx, input_mask \
            = process_batch(batch_question, reverse=options['reverse'])
        batch_image_feat = reshape_image_feat(batch_image_feat,
                                              options['num_region'],
                                              options['region_dim'])

        # do the training process!!!
        _, loss ,aucc = sess.run(
                [train_op, tf_loss, tf_aucc],
                feed_dict={
                    tf_image: batch_image_feat,
                    tf_question: input_idx,
                    tf_label: batch_answer_label.astype('int32').flatten()
                    })

        current_learning_rate = lr*decay_factor
        lr.assign(current_learning_rate).eval(session=sess)

if __name__ == '__main__':
    
    with tf.device('/gpu:'+str(options['gpu_id'])):
        train(options)