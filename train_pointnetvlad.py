import argparse
import math
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import logging
import logging.config
from pointnetvlad_cls import *
from loading_pointclouds import *
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree

from config import *
from data.datagenerator import DataGenerator
from models.net_factory import get_network
from utils import get_tensors_in_checkpoint_file
from pprint import pprint

CKPT_PATH = './ckpt/secondstage/ckpt/checkpoint.ckpt-210000'

#params
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1, help='GPU to use [default: GPU 1]')
parser.add_argument('--log_dir', default='log/', help='Log dir [default: log]')
parser.add_argument('--positives_per_query', type=int, default=2, help='Number of potential positives in each training tuple [default: 2]')
parser.add_argument('--negatives_per_query', type=int, default=18, help='Number of definite negatives in each training tuple [default: 18]')
parser.add_argument('--max_epoch', type=int, default=20, help='Epoch to run [default: 20]')
parser.add_argument('--batch_num_queries', type=int, default=2, help='Batch Size during training [default: 2]')
parser.add_argument('--learning_rate', type=float, default=0.00005, help='Initial learning rate [default: 0.00005]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--margin_1', type=float, default=0.5, help='Margin for hinge loss [default: 0.5]')
parser.add_argument('--margin_2', type=float, default=0.2, help='Margin for hinge loss [default: 0.2]')
parser.add_argument('--pretrained', type=str, default=CKPT_PATH, help='Pretrained model path for transfer learning')
parser.add_argument('--base_scale', type=float, default=2.0,
                    help='Radius for sampling clusters (default: 2.0)')
parser.add_argument('--num_samples', type=int, default=64,
                    help='Maximum number of points to consider per cluster (default: 64)')
parser.add_argument('--feature_dim', type=int, default=32, choices=[16, 32, 64, 128],
                    help='Feature dimension size (default: 32)')
parser.add_argument('--data_dim', type=int, default=6,
                    help='Input dimension for data. Note: Feat3D-Net will only use the first 3 dimensions (default: 6)')
parser.add_argument('--transfer_learning', type=bool, default=False, help='Set to true if using a pretrained model, else false if training from scratch')
parser.add_argument('--model', type=str, default='3DFeatNet',
                    help='Model to load')
FLAGS = parser.parse_args()

# Create Logging
logging.config.fileConfig('logging.conf')
logger = logging.getLogger(__name__)

BATCH_NUM_QUERIES = FLAGS.batch_num_queries
EVAL_BATCH_SIZE = 1
NUM_POINTS = 4096
POSITIVES_PER_QUERY = FLAGS.positives_per_query
NEGATIVES_PER_QUERY = FLAGS.negatives_per_query
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
MARGIN1 = FLAGS.margin_1
MARGIN2 = FLAGS.margin_2

TRAIN_FILE = 'generating_queries/training_queries_baseline.pickle'
TEST_FILE = 'generating_queries/test_queries_baseline.pickle'

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

#Load dictionary of training queries
TRAINING_QUERIES = get_queries_dict(TRAIN_FILE)
TEST_QUERIES = get_queries_dict(TEST_FILE)

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

global HARD_NEGATIVES
HARD_NEGATIVES={}

global TRAINING_LATENT_VECTORS
TRAINING_LATENT_VECTORS=[]

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_NUM_QUERIES,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

#learning rate halfed every 5 epoch
def get_learning_rate(epoch):
    learning_rate = BASE_LEARNING_RATE*((0.9)**(epoch//5))
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

def initialize_model(sess, checkpoint, ignore_missing_vars=False, restore_exclude=None):
    print('Initializing weights')

    sess.run(tf.global_variables_initializer())

    if checkpoint is not None:
        print('Restoring model from {}'.format(checkpoint))

        model_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        print(tf.GraphKeys.GLOBAL_VARIABLES)
        exclude_list = []
        if restore_exclude is not None:
            for e in restore_exclude:
                exclude_list += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=e)
        for e in exclude_list:
            print('Excluded from model restore: %s', e.op.name)

        if ignore_missing_vars:
            checkpoint_var_names = get_tensors_in_checkpoint_file(checkpoint)
            missing = [m.op.name for m in model_var_list if m.op.name not in checkpoint_var_names and m not in exclude_list]

            for m in missing:
                print('Variable missing from checkpoint: %s', m)

            var_list = [m for m in model_var_list if m.op.name in checkpoint_var_names and m not in exclude_list]

        else:
            var_list = [m for m in model_var_list if m not in exclude_list]

        pprint(var_list)

        saver = tf.train.Saver(var_list)

        saver.restore(sess, checkpoint)

def transfer_learning(descriptors):
    global HARD_NEGATIVES

    with tf.Session(config=config) as sess:
        with tf.device('/gpu:'+str(GPU_INDEX)):
            initialize_model(sess, CKPT_PATH)

            print("In Graph")

            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)

            batch = tf.Variable(0)
            epoch_num = tf.placeholder(tf.float32, shape=())
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            with tf.variable_scope("query_triplets") as scope:
                out_vecs = forward_netvlad(descriptors, is_training_pl, bn_decay=bn_decay)
                print(out_vecs)
                q_vec, pos_vecs, neg_vecs, other_neg_vec = tf.split(out_vecs, [1,POSITIVES_PER_QUERY, NEGATIVES_PER_QUERY,1],1)
                print(q_vec)
                print(pos_vecs)
                print(neg_vecs)
                print(other_neg_vec)

            #loss = lazy_triplet_loss(q_vec, pos_vecs, neg_vecs, MARGIN1)
            #loss = softmargin_loss(q_vec, pos_vecs, neg_vecs)
            #loss = quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg_vec, MARGIN1, MARGIN2)
            loss = lazy_quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg_vec, MARGIN1, MARGIN2)
            tf.summary.scalar('loss', loss)

            # Get training operator
            learning_rate = get_learning_rate(epoch_num)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss, global_step=batch)

            # Add summary writers
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                    sess.graph)
            test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

            # Initialize a new model
            init = tf.global_variables_initializer()
            sess.run(init)
            print("Initialized")

            # Restore a model
            # saver.restore(sess, os.path.join(LOG_DIR, "model.ckpt"))
            # print("Model restored.")

            ops = {'query': query,
                'positives': positives,
                'negatives': negatives,
                'other_negatives': other_negatives,
                'is_training_pl': is_training_pl,
                'loss': loss,
                'train_op': train_op,
                'merged': merged,
                'step': batch,
                'epoch_num': epoch_num,
                'q_vec':q_vec,
                'pos_vecs': pos_vecs,
                'neg_vecs': neg_vecs,
                'other_neg_vec': other_neg_vec}


            for epoch in range(MAX_EPOCH):
                print(epoch)
                print()
                log_string('**** EPOCH %03d ****' % (epoch))
                sys.stdout.flush()

                train_one_epoch(sess, ops, train_writer, test_writer, epoch, saver)

def train():
    global HARD_NEGATIVES
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            print("In Graph")
            query = placeholder_inputs(BATCH_NUM_QUERIES, 1, NUM_POINTS)
            positives = placeholder_inputs(BATCH_NUM_QUERIES, POSITIVES_PER_QUERY, NUM_POINTS)
            negatives = placeholder_inputs(BATCH_NUM_QUERIES, NEGATIVES_PER_QUERY, NUM_POINTS)
            other_negatives = placeholder_inputs(BATCH_NUM_QUERIES,1, NUM_POINTS)

            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)

            batch = tf.Variable(0)
            epoch_num = tf.placeholder(tf.float32, shape=())
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            with tf.variable_scope("query_triplets") as scope:
                vecs = tf.concat([query, positives, negatives, other_negatives],1)
                print(vecs)
                out_vecs = forward(vecs, is_training_pl, bn_decay=bn_decay)
                print(out_vecs)
                q_vec, pos_vecs, neg_vecs, other_neg_vec= tf.split(out_vecs, [1,POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY,1],1)
                print(q_vec)
                print(pos_vecs)
                print(neg_vecs)
                print(other_neg_vec)

            #loss = lazy_triplet_loss(q_vec, pos_vecs, neg_vecs, MARGIN1)
            #loss = softmargin_loss(q_vec, pos_vecs, neg_vecs)
            #loss = quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg_vec, MARGIN1, MARGIN2)
            loss = lazy_quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg_vec, MARGIN1, MARGIN2)
            tf.summary.scalar('loss', loss)

            # Get training operator
            learning_rate = get_learning_rate(epoch_num)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False

        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Initialize a new model
        init = tf.global_variables_initializer()
        sess.run(init)
        print("Initialized")

        # Restore a model
        saver.restore(sess, FLAGS.pretrained)
        print("Model restored.")


        ops = {'query': query,
               'positives': positives,
               'negatives': negatives,
               'other_negatives': other_negatives,
               'is_training_pl': is_training_pl,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'epoch_num': epoch_num,
               'q_vec':q_vec,
               'pos_vecs': pos_vecs,
               'neg_vecs': neg_vecs,
               'other_neg_vec': other_neg_vec}


        for epoch in range(MAX_EPOCH):
            print(epoch)
            print()
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer, test_writer, epoch, saver)



def train_one_epoch(sess, ops, train_writer, test_writer, epoch, saver):
    global HARD_NEGATIVES
    global TRAINING_LATENT_VECTORS

    is_training = True
    sampled_neg=4000
    #number of hard negatives in the training tuple
    #which are taken from the sampled negatives
    num_to_take=10

    # Shuffle train files
    train_file_idxs = np.arange(0, len(TRAINING_QUERIES.keys()))
    np.random.shuffle(train_file_idxs)

    for i in range(len(train_file_idxs)//BATCH_NUM_QUERIES):
        batch_keys= train_file_idxs[i*BATCH_NUM_QUERIES:(i+1)*BATCH_NUM_QUERIES]
        q_tuples=[]

        faulty_tuple=False
        no_other_neg=False
        for j in range(BATCH_NUM_QUERIES):
            if(len(TRAINING_QUERIES[batch_keys[j]]["positives"])<POSITIVES_PER_QUERY):
                faulty_tuple=True
                break

            #no cached feature vectors
            if(len(TRAINING_LATENT_VECTORS)==0):
                q_tuples.append(get_query_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_neg=[], other_neg=True))
                # q_tuples.append(get_rotated_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_neg=[], other_neg=True))
                # q_tuples.append(get_jittered_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_neg=[], other_neg=True))

            elif(len(HARD_NEGATIVES.keys())==0):
                query=get_feature_representation(TRAINING_QUERIES[batch_keys[j]]['query'], sess, ops)
                random.shuffle(TRAINING_QUERIES[batch_keys[j]]['negatives'])
                negatives=TRAINING_QUERIES[batch_keys[j]]['negatives'][0:sampled_neg]
                hard_negs= get_random_hard_negatives(query, negatives, num_to_take)
                print(hard_negs)
                q_tuples.append(get_query_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True))
                # q_tuples.append(get_rotated_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True))
                # q_tuples.append(get_jittered_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True))
            else:
                query=get_feature_representation(TRAINING_QUERIES[batch_keys[j]]['query'], sess, ops)
                random.shuffle(TRAINING_QUERIES[batch_keys[j]]['negatives'])
                negatives=TRAINING_QUERIES[batch_keys[j]]['negatives'][0:sampled_neg]
                hard_negs= get_random_hard_negatives(query, negatives, num_to_take)
                hard_negs= list(set().union(HARD_NEGATIVES[batch_keys[j]], hard_negs))
                print('hard',hard_negs)
                q_tuples.append(get_query_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True))
                # q_tuples.append(get_rotated_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True))
                # q_tuples.append(get_jittered_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True))

            if(q_tuples[j][3].shape[0]!=NUM_POINTS):
                no_other_neg= True
                break

        #construct query array
        if(faulty_tuple):
            log_string('----' + str(i) + '-----')
            log_string('----' + 'FAULTY TUPLE' + '-----')
            continue

        if(no_other_neg):
            log_string('----' + str(i) + '-----')
            log_string('----' + 'NO OTHER NEG' + '-----')
            continue

        queries=[]
        positives=[]
        negatives=[]
        other_neg=[]
        for k in range(len(q_tuples)):
            queries.append(q_tuples[k][0])
            positives.append(q_tuples[k][1])
            negatives.append(q_tuples[k][2])
            other_neg.append(q_tuples[k][3])

        queries= np.array(queries)
        queries= np.expand_dims(queries,axis=1)
        other_neg= np.array(other_neg)
        other_neg= np.expand_dims(other_neg,axis=1)
        positives= np.array(positives)
        negatives= np.array(negatives)
        log_string('----' + str(i) + '-----')
        if(len(queries.shape)!=4):
            log_string('----' + 'FAULTY QUERY' + '-----')
            continue

        feed_dict={ops['query']:queries, ops['positives']:positives, ops['negatives']:negatives, ops['other_negatives']:other_neg, ops['is_training_pl']:is_training, ops['epoch_num']:epoch}
        summary, step, train, loss_val = sess.run([ops['merged'], ops['step'],
                ops['train_op'], ops['loss']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        log_string('batch loss: %f' % loss_val)

        if(i%200==7):
            test_file_idxs = np.arange(0,len(TEST_QUERIES.keys()))
            np.random.shuffle(test_file_idxs)

            eval_loss=0
            eval_batches=5
            eval_batches_counted=0
            for eval_batch in range(eval_batches):
                eval_keys= test_file_idxs[eval_batch*BATCH_NUM_QUERIES:(eval_batch+1)*BATCH_NUM_QUERIES]
                eval_tuples=[]

                faulty_eval_tuple=False
                no_other_neg= False
                for e_tup in range(BATCH_NUM_QUERIES):
                    if(len(TEST_QUERIES[eval_keys[e_tup]]["positives"])<POSITIVES_PER_QUERY):
                        faulty_eval_tuple=True
                        break
                    eval_tuples.append(get_query_tuple(TEST_QUERIES[eval_keys[e_tup]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TEST_QUERIES, hard_neg=[], other_neg=True))

                    if(eval_tuples[e_tup][3].shape[0]!=NUM_POINTS):
                        no_other_neg= True
                        break

                if(faulty_eval_tuple):
                    log_string('----' + 'FAULTY EVAL TUPLE' + '-----')
                    continue

                if(no_other_neg):
                    log_string('----' + str(i) + '-----')
                    log_string('----' + 'NO OTHER NEG EVAL' + '-----')
                    continue

                eval_batches_counted+=1
                eval_queries=[]
                eval_positives=[]
                eval_negatives=[]
                eval_other_neg=[]

                for tup in range(len(eval_tuples)):
                    eval_queries.append(eval_tuples[tup][0])
                    eval_positives.append(eval_tuples[tup][1])
                    eval_negatives.append(eval_tuples[tup][2])
                    eval_other_neg.append(eval_tuples[tup][3])

                eval_queries= np.array(eval_queries)
                eval_queries= np.expand_dims(eval_queries,axis=1)
                eval_other_neg= np.array(eval_other_neg)
                eval_other_neg= np.expand_dims(eval_other_neg,axis=1)
                eval_positives= np.array(eval_positives)
                eval_negatives= np.array(eval_negatives)
                feed_dict={ops['query']:eval_queries, ops['positives']:eval_positives, ops['negatives']:eval_negatives, ops['other_negatives']:eval_other_neg, ops['is_training_pl']:False, ops['epoch_num']:epoch}
                e_summary, e_step, e_loss= sess.run([ops['merged'], ops['step'], ops['loss']], feed_dict=feed_dict)
                eval_loss+=e_loss
                if(eval_batch==4):
                    test_writer.add_summary(e_summary, e_step)
            average_eval_loss= float(eval_loss)/eval_batches_counted
            log_string('\t\t\tEVAL')
            log_string('\t\t\teval_loss: %f' %average_eval_loss)


        if(epoch>5 and i%700 ==29):
            #update cached feature vectors
            TRAINING_LATENT_VECTORS=get_latent_vectors(sess, ops, TRAINING_QUERIES)
            print("Updated cached feature vectors")

        if(i%3000==101):
            save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
            log_string("Model saved in file: %s" % save_path)


def get_feature_representation(filename, sess, ops):
    is_training=False
    queries=load_pc_files([filename])
    queries= np.expand_dims(queries,axis=1)
    if(BATCH_NUM_QUERIES-1>0):
        fake_queries=np.zeros((BATCH_NUM_QUERIES-1,1,NUM_POINTS,3))
        q=np.vstack((queries,fake_queries))
    else:
        q=queries
    fake_pos=np.zeros((BATCH_NUM_QUERIES,POSITIVES_PER_QUERY,NUM_POINTS,3))
    fake_neg=np.zeros((BATCH_NUM_QUERIES,NEGATIVES_PER_QUERY,NUM_POINTS,3))
    fake_other_neg=np.zeros((BATCH_NUM_QUERIES,1,NUM_POINTS,3))
    feed_dict={ops['query']:q, ops['positives']:fake_pos, ops['negatives']:fake_neg, ops['other_negatives']: fake_other_neg, ops['is_training_pl']:is_training}
    output=sess.run(ops['q_vec'], feed_dict=feed_dict)
    output=output[0]
    output=np.squeeze(output)
    return output

def get_random_hard_negatives(query_vec, random_negs, num_to_take):
    global TRAINING_LATENT_VECTORS

    latent_vecs=[]
    for j in range(len(random_negs)):
        latent_vecs.append(TRAINING_LATENT_VECTORS[random_negs[j]])

    latent_vecs=np.array(latent_vecs)
    nbrs = KDTree(latent_vecs)
    distances, indices = nbrs.query(np.array([query_vec]),k=num_to_take)
    hard_negs=np.squeeze(np.array(random_negs)[indices[0]])
    hard_negs= hard_negs.tolist()
    return hard_negs

def get_latent_vectors(sess, ops, dict_to_process):
    is_training=False
    train_file_idxs = np.arange(0, len(dict_to_process.keys()))

    batch_num= BATCH_NUM_QUERIES*(1+POSITIVES_PER_QUERY+NEGATIVES_PER_QUERY+1)
    q_output = []
    for q_index in range(len(train_file_idxs)//batch_num):
        file_indices=train_file_idxs[q_index*batch_num:(q_index+1)*(batch_num)]
        file_names=[]
        for index in file_indices:
            file_names.append(dict_to_process[index]["query"])
        queries=load_pc_files(file_names)

        q1=queries[0:BATCH_NUM_QUERIES]
        q1=np.expand_dims(q1,axis=1)

        q2=queries[BATCH_NUM_QUERIES:BATCH_NUM_QUERIES*(POSITIVES_PER_QUERY+1)]
        q2=np.reshape(q2,(BATCH_NUM_QUERIES,POSITIVES_PER_QUERY,NUM_POINTS,3))

        q3=queries[BATCH_NUM_QUERIES*(POSITIVES_PER_QUERY+1):BATCH_NUM_QUERIES*(NEGATIVES_PER_QUERY+POSITIVES_PER_QUERY+1)]
        q3=np.reshape(q3,(BATCH_NUM_QUERIES,NEGATIVES_PER_QUERY,NUM_POINTS,3))

        q4=queries[BATCH_NUM_QUERIES*(NEGATIVES_PER_QUERY+POSITIVES_PER_QUERY+1):BATCH_NUM_QUERIES*(NEGATIVES_PER_QUERY+POSITIVES_PER_QUERY+2)]
        q4=np.expand_dims(q4,axis=1)

        feed_dict={ops['query']:q1, ops['positives']:q2, ops['negatives']:q3,ops['other_negatives']:q4, ops['is_training_pl']:is_training}
        o1, o2, o3, o4=sess.run([ops['q_vec'], ops['pos_vecs'], ops['neg_vecs'], ops['other_neg_vec']], feed_dict=feed_dict)

        o1=np.reshape(o1,(-1,o1.shape[-1]))
        o2=np.reshape(o2,(-1,o2.shape[-1]))
        o3=np.reshape(o3,(-1,o3.shape[-1]))
        o4=np.reshape(o4,(-1,o4.shape[-1]))

        out=np.vstack((o1,o2,o3,o4))
        q_output.append(out)

    q_output=np.array(q_output)
    if(len(q_output)!=0):
        q_output=q_output.reshape(-1,q_output.shape[-1])

    #handle edge case
    for q_index in range((len(train_file_idxs)//batch_num*batch_num),len(dict_to_process.keys())):
        index=train_file_idxs[q_index]
        queries=load_pc_files([dict_to_process[index]["query"]])
        queries= np.expand_dims(queries,axis=1)

        if(BATCH_NUM_QUERIES-1>0):
            fake_queries=np.zeros((BATCH_NUM_QUERIES-1,1,NUM_POINTS,3))
            q=np.vstack((queries,fake_queries))
        else:
            q=queries

        fake_pos=np.zeros((BATCH_NUM_QUERIES,POSITIVES_PER_QUERY,NUM_POINTS,3))
        fake_neg=np.zeros((BATCH_NUM_QUERIES,NEGATIVES_PER_QUERY,NUM_POINTS,3))
        fake_other_neg=np.zeros((BATCH_NUM_QUERIES,1,NUM_POINTS,3))
        feed_dict={ops['query']:q, ops['positives']:fake_pos, ops['negatives']:fake_neg, ops['other_negatives']:fake_other_neg, ops['is_training_pl']:is_training}
        output=sess.run(ops['q_vec'], feed_dict=feed_dict)
        output=output[0]
        output=np.squeeze(output)
        if (q_output.shape[0]!=0):
            q_output=np.vstack((q_output,output))
        else:
            q_output=output

    print(q_output.shape)
    return q_output

def compute_descriptors():

    log_arguments()

    logger.debug('In compute_descriptors()')
    logger.info('Computed descriptors will be saved to %s', args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # Data
    binFiles = [f for f in os.listdir(args.data_dir) if f.endswith('.bin')]
    data_dim = args.data_dim
    logger.info('Found %i bin files in directory: %s, each assumed to be of dim %i',
                len(binFiles), args.data_dir, data_dim)

    # Model
    param = {'NoRegress': False, 'BaseScale': args.base_scale, 'Attention': True,
             'num_clusters': -1, 'num_samples': args.num_samples, 'feature_dim': args.feature_dim}
    model = get_network(args.model)(param)

    # placeholders
    is_training = tf.placeholder(tf.bool)
    cloud_pl, _, _ = model.get_placeholders(data_dim)

    # Ops1
    xyz_op, features_op, attention_op, end_points = model.get_inference_model(cloud_pl, is_training, use_bn=USE_BN)

    with tf.Session(config=config) as sess:

        initialize_model(sess, args.checkpoint)

        num_processed = 0

        # Training data
        for iBin in range(0, len(binFiles)):

            binFile = binFiles[iBin]
            fname_no_ext = binFile[:-4]
            pointcloud = DataGenerator.load_point_cloud(os.path.join(args.data_dir, binFile), num_cols=data_dim)

            if args.randomize_points:
                permutation = np.random.choice(pointcloud.shape[0], size=pointcloud.shape[0], replace=False)
                inv_permutation = np.zeros_like(permutation)
                inv_permutation[permutation] = range(0, pointcloud.shape[0])
                pointcloud = pointcloud[permutation, :]
            else:
                inv_permutation = np.arange(0, pointcloud.shape[0], dtype=np.int64)
            if args.num_points > 0:
                pointcloud = pointcloud[:args.num_points, :]

            pointclouds = pointcloud[None, :, :]
            num_models = pointclouds.shape[0]

            if args.use_keypoints_from is None:
                # Detect features

                # Compute attention in batches due to limited memory
                xyz, attention = [], []
                for startPt in range(0, pointcloud.shape[0], MAX_POINTS):
                    endPt = min(pointcloud.shape[0], startPt + MAX_POINTS)
                    xyz_subset = pointclouds[:, startPt:endPt, :3]

                    # Compute attention over all points
                    xyz_cur, attention_cur = \
                        sess.run([xyz_op, attention_op],
                                 feed_dict={cloud_pl: pointclouds, is_training: False,
                                            end_points['keypoints']: xyz_subset})

                    xyz.append(xyz_cur)
                    attention.append(attention_cur)

                xyz = np.concatenate(xyz, axis=1)
                attention = np.concatenate(attention, axis=1)

                # # Uncomment to save out attention to file
                # with open(os.path.join(args.output_dir, '{}_attention.bin'.format(fname_no_ext)), 'wb') as f:
                #     if args.num_points > 0:
                #         xyz_attention = np.concatenate((xyz[0, :, :],
                #                                         np.expand_dims(attention[0, :], 1),), axis=1)
                #     else:
                #         xyz_attention = np.concatenate((xyz[0, inv_permutation, :],
                #                                         np.expand_dims(attention[0, inv_permutation], 1),), axis=1)
                #     xyz_attention.tofile(f)

                # Non maximal suppression to select keypoints based on attention
                xyz_nms, attention_nms, num_keypoints = nms(xyz, attention)

            else:
                # Load keypoints from file
                xyz_nms = []
                for i in range(num_models):
                    kp_fname = os.path.join(args.use_keypoints_from, '{}_kp.bin'.format(fname_no_ext))
                    xyz_nms.append(DataGenerator.load_point_cloud(kp_fname, num_cols=3))

                # Pad to make same size
                num_keypoints = [kp.shape[0] for kp in xyz_nms]
                largest_kp_count = max(num_keypoints)
                for i in range(num_models):
                    num_to_pad = largest_kp_count-xyz_nms[i].shape[0]
                    to_pad_with = np.repeat(xyz_nms[i][0,:][None, :], num_to_pad, axis=0)
                    xyz_nms[i] = np.concatenate((xyz_nms[i], to_pad_with), axis=0)
                xyz_nms = np.stack(xyz_nms, axis=0)

            # Compute features
            xyz, features = \
                sess.run([xyz_op, features_op],
                         feed_dict={cloud_pl: pointclouds, is_training: False, end_points['keypoints']: xyz_nms})

            # Save out the output
            with open(os.path.join(args.output_dir, '{}.bin'.format(fname_no_ext)), 'wb') as f:
                xyz_features = np.concatenate([xyz[0, 0:num_keypoints[0], :], features[0, 0:num_keypoints[0], :]],
                                              axis=1)
                xyz_features.tofile(f)

            num_processed += 1
            logger.info('Processed %i / %i images', num_processed, len(binFiles))


def initialize_model(sess, checkpoint, ignore_missing_vars=False, restore_exclude=None):
    logger.info('Initializing weights')

    sess.run(tf.global_variables_initializer())

    if checkpoint is not None:

        logger.info('Restoring model from {}'.format(args.checkpoint))

        model_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        exclude_list = []
        if restore_exclude is not None:
            for e in restore_exclude:
                exclude_list += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=e)
        for e in exclude_list:
            logger.info('Excluded from model restore: %s', e.op.name)

        if ignore_missing_vars:
            checkpoint_var_names = get_tensors_in_checkpoint_file(checkpoint)
            missing = [m.op.name for m in model_var_list if m.op.name not in checkpoint_var_names and m not in exclude_list]

            for m in missing:
                logger.warning('Variable missing from checkpoint: %s', m)

            var_list = [m for m in model_var_list if m.op.name in checkpoint_var_names and m not in exclude_list]

        else:
            var_list = [m for m in model_var_list if m not in exclude_list]

        print('Var list: {}'.format(var_list))
        saver = tf.train.Saver(var_list)

        saver.restore(sess, checkpoint)

    logger.info('Weights initialized')


def log_arguments():
    s = '\n'.join(['    {}: {}'.format(arg, getattr(args, arg)) for arg in vars(args)])
    s = 'Arguments:\n' + s
    logger.info(s)


def nms(xyz, attention):

    num_models = xyz.shape[0]  # Should be equals to batch size
    num_keypoints = [0] * num_models

    xyz_nms = np.zeros((num_models, args.max_keypoints, 3), xyz.dtype)
    attention_nms = np.zeros((num_models, args.max_keypoints), xyz.dtype)

    for i in range(num_models):

        nbrs = NearestNeighbors(n_neighbors=50, algorithm='ball_tree').fit(xyz[i, :, :])
        distances, indices = nbrs.kneighbors(xyz[i, :, :])

        knn_attention = attention[i, indices]
        outside_ball = distances > args.nms_radius
        knn_attention[outside_ball] = 0.0
        is_max = np.where(np.argmax(knn_attention, axis=1) == 0)[0]

        # Extract the top k features, filtering out weak responses
        attention_thresh = np.max(attention[i, :]) * args.min_response_ratio
        is_max_attention = [(attention[i, m], m) for m in is_max if attention[i, m] > attention_thresh]
        is_max_attention = sorted(is_max_attention, reverse=True)
        max_indices = [m[1] for m in is_max_attention]

        if len(max_indices) >= args.max_keypoints:
            max_indices = max_indices[:args.max_keypoints]
            num_keypoints[i] = len(max_indices)
        else:
            num_keypoints[i] = len(max_indices)  # Retrain original number of points
            max_indices = np.pad(max_indices, (0, args.max_keypoints - len(max_indices)), 'constant',
                                 constant_values=max_indices[0])

        xyz_nms[i, :, :] = xyz[i, max_indices, :]
        attention_nms[i, :] = attention[i, max_indices]

    return xyz_nms, attention_nms, num_keypoints

if __name__ == "__main__":
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False

    if FLAGS.transfer_learning:
        with tf.device('/CPU:0'):
            descriptors = compute_descriptors()

        transfer_learning(descriptors)

    else:
        train()
