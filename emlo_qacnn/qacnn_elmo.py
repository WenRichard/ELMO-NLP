# -*- coding: utf-8 -*-
# @Time    : 2019/3/19 20:19
# @Author  : Alan
# @Email   : xiezhengwen2013@163.com
# @File    : train.py
# @Software: PyCharm


import time
import logging
import os
import sys
from copy import deepcopy
stdout = sys.stdout

from data_helper_emlo import *
from model_elmo import SiameseQACNN_elmo
from model_utils import *
from bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers, Batcher

# 创建一个logger
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)

# 创建一个handler，用于写入日志文件
timestamp = str(int(time.time()))
fh = logging.FileHandler('./log/log_' + timestamp +'.txt')
fh.setLevel(logging.DEBUG)

# 定义handler的输出格式
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
fh.setFormatter(formatter)
# ch.setFormatter(formatter)

# 给logger添加handler
logger.addHandler(fh)
# logger.addHandler(ch)


class NNConfig(object):
    def __init__(self, embeddings=None):
        # 设置elmo参数
        self.optionFile = "modelParams/elmo_options.json"
        self.weightFile = "modelParams/elmo_weights.hdf5"
        self.vocabFile = "modelParams/vocab.txt"
        self.tokenEmbeddingFile = 'modelParams/elmo_token_embeddings.hdf5'
        self.embedding_size = 256
        # 输入问题(句子)长度
        self.ques_length = 25
        # 输入答案长度
        self.ans_length = 90
        # 循环数
        self.num_epochs = 30
        # batch大小
        self.batch_size = 128
        # 不同类型的filter，对应不同的尺寸
        self.window_sizes = [1, 2, 3, 5, 7, 9]
        # 隐层大小
        self.hidden_size = 128
        self.output_size = 128
        self.keep_prob = 0.5
        # 每种filter的数量
        self.n_filters = 128
        # margin大小
        self.margin = 0.5
        # 词向量大小
        self.embeddings = np.array(embeddings).astype(np.float32)
        # 学习率
        self.learning_rate = 0.001
        # contrasive loss 中的 positive loss部分的权重
        self.pos_weight = 0.25
        # 优化器
        self.optimizer = 'adam'
        self.clip_value = 5
        self.l2_lambda = 0.0001
        # 评测
        self.eval_batch = 100


def evaluate(sess, model, corpus, elmo, config):
    iterator = Iterator(corpus)

    count = 0
    total_qids = []
    total_aids = []
    total_pred = []
    total_labels = []
    total_loss = 0.
    Acc = []
    for batch_x in iterator.next(config.batch_size, shuffle=False):
        batch_qids, batch_q, batch_aids, batch_a, batch_qmask, batch_amask, labels = zip(*batch_x)
        batch_q = np.asarray(batch_q)
        batch_a = np.asarray(batch_a)
        q_ap_cosine, loss, acc = sess.run([model.q_a_cosine, model.total_loss, model.accu],
                                     feed_dict={model._ques: elmo(batch_q)[0],
                                                model._ans: elmo(batch_a)[0],
                                                model._ans_neg: elmo(batch_a)[0],
                                                model.dropout_keep_prob: 1.0})
        total_loss += loss
        Acc.append(acc)
        count += 1
        total_qids.append(batch_qids)
        total_aids.append(batch_aids)
        total_pred.append(q_ap_cosine)
        total_labels.append(labels)

        # print(batch_qids[0], [id2word[_] for _ in batch_q[0]],
        #     batch_aids[0], [id2word[_] for _ in batch_ap[0]])
    total_qids = np.concatenate(total_qids, axis=0)
    total_aids = np.concatenate(total_aids, axis=0)
    total_pred = np.concatenate(total_pred, axis=0)
    total_labels = np.concatenate(total_labels, axis=0)
    MAP, MRR = eval_map_mrr(total_qids, total_aids, total_pred, total_labels)
    acc_ = np.sum(Acc)/count
    ave_loss = total_loss/count
    # print('Eval loss:{}'.format(total_loss / count))
    return MAP, MRR, ave_loss, acc_


def test(corpus, elmo, config):
    with tf.Session() as sess:
        model = SiameseQACNN_elmo(config)
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(best_path))
        test_MAP, test_MRR, _, acc = evaluate(sess, model, corpus, elmo, config)
        print('start test...............')
        print("-- test MAP %.5f -- test MRR %.5f" % (test_MAP, test_MRR))


def train(train_corpus, val_corpus, test_corpus, config, eval_train_corpus=None):
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9  # 配置gpu占用率

        sess = tf.Session(config=session_conf)
        iterator = Iterator(train_corpus)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(best_path):
            os.makedirs(best_path)

        with sess.as_default():
            # training
            print('Start training and evaluating ...')
            start_time = time.time()

            model = SiameseQACNN_elmo(config)
            # 实例化BiLM对象，这个必须放置在全局下，不能在elmo函数中定义，否则会出现重复生成tensorflow节点。
            with tf.variable_scope("bilm", reuse=True):
                bilm = BidirectionalLanguageModel(
                    config.optionFile,
                    config.weightFile,
                    use_character_inputs=False,
                    embedding_weight_file=config.tokenEmbeddingFile
                )
            inputData = tf.placeholder('int32', shape=(None, None))
            # 调用bilm中的__call__方法生成op对象
            inputEmbeddingsOp = bilm(inputData)

            # 计算ELMo向量表示
            elmoInput = weight_layers('input', inputEmbeddingsOp, l2_coef=0.0)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
            best_saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
            ckpt = tf.train.get_checkpoint_state(save_path)
            print('Configuring TensorBoard and Saver ...')
            summary_writer = tf.summary.FileWriter(save_path, graph=sess.graph)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print('Reloading model parameters..')
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('Created new model parameters..')
                sess.run(tf.global_variables_initializer())

            # count trainable parameters
            total_parameters = count_parameters()
            print('Total trainable parameters : {}'.format(total_parameters))

            def elmo(reviews):
                """
                对每一个输入的batch都动态的生成词向量表示
                """

                #           tf.reset_default_graph()
                # TokenBatcher是生成词表示的batch类
                batcher = TokenBatcher(config.vocabFile)
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())

                    # 生成batch数据
                    inputDataIndex = batcher.batch_sentences(reviews)

                    # 计算ELMo的向量表示
                    elmoInputVec = sess.run(
                        [elmoInput['weighted_op']],
                        feed_dict={inputData: inputDataIndex}
                    )

                return elmoInputVec

            current_step = 0
            best_map_val = 0.0
            best_mrr_val = 0.0
            last_dev_map = 0.0
            last_dev_mrr = 0.0
            for epoch in range(config.num_epochs):
                print("----- Epoch {}/{} -----".format(epoch + 1, config.num_epochs))
                count = 0
                for batch_x in iterator.next(config.batch_size, shuffle=True):

                    batch_q, batch_a_pos, batch_a_neg, batch_qmask, batch_a_pos_mask, batch_a_neg_mask = zip(*batch_x)
                    batch_q = np.asarray(batch_q)
                    batch_a_pos = np.asarray(batch_a_pos)
                    batch_a_neg = np.asarray(batch_a_neg)
                    # print('s', np.shape(elmo(batch_q)[0]))
                    _, loss, summary, train_acc = sess.run([model.train_op, model.total_loss, model.summary_op, model.accu],
                                                        feed_dict={model._ques: elmo(batch_q)[0],
                                                                  model._ans: elmo(batch_a_pos)[0],
                                                                  model._ans_neg: elmo(batch_a_neg)[0],
                                                                  model.dropout_keep_prob: config.keep_prob})
                    count += 1
                    current_step += 1
                    if count % 10 == 0:
                        print('[epoch {}, batch {}]Loss:{}'.format(epoch, count, loss))
                    summary_writer.add_summary(summary, current_step)
                if eval_train_corpus is not None:
                    train_MAP, train_MRR, train_Loss, train_acc_ = evaluate(sess, model, eval_train_corpus, elmo, config)
                    print("--- epoch %d  -- train Loss %.5f -- train Acc %.5f -- train MAP %.5f -- train MRR %.5f" % (
                            epoch+1, train_Loss, train_acc_, train_MAP, train_MRR))
                if val_corpus is not None:
                    dev_MAP, dev_MRR, dev_Loss, dev_acc = evaluate(sess, model, val_corpus, elmo, config)
                    print("--- epoch %d  -- dev Loss %.5f -- dev Acc %.5f --dev MAP %.5f -- dev MRR %.5f" % (
                        epoch + 1, dev_Loss, dev_acc, dev_MAP, dev_MRR))
                    logger.info("\nEvaluation:")
                    logger.info("--- epoch %d  -- dev Loss %.5f -- dev Acc %.5f --dev MAP %.5f -- dev MRR %.5f" % (
                        epoch + 1, dev_Loss, dev_acc, dev_MAP, dev_MRR))

                    test_MAP, test_MRR, test_Loss, test_acc= evaluate(sess, model, test_corpus, elmo, config)
                    print("--- epoch %d  -- test Loss %.5f -- test Acc %.5f --test MAP %.5f -- test MRR %.5f" % (
                        epoch + 1, test_Loss, test_acc, test_MAP, test_MRR))
                    logger.info("\nTest:")
                    logger.info("--- epoch %d  -- test Loss %.5f -- dev Acc %.5f --test MAP %.5f -- test MRR %.5f" % (
                        epoch + 1, test_Loss, test_acc, test_MAP, test_MRR))

                    checkpoint_path = os.path.join(save_path, 'map{:.5f}_{}.ckpt'.format(test_MAP, current_step))
                    bestcheck_path = os.path.join(best_path, 'map{:.5f}_{}.ckpt'.format(test_MAP, current_step))
                    saver.save(sess, checkpoint_path, global_step=epoch)
                    if test_MAP > best_map_val or test_MRR > best_mrr_val:
                        best_map_val = test_MAP
                        best_mrr_val = test_MRR
                        best_saver.save(sess, bestcheck_path, global_step=epoch)
                    last_dev_map = test_MAP
                    last_dev_mrr = test_MRR
            logger.info("\nBest and Last:")
            logger.info('--- best_MAP %.4f -- best_MRR %.4f -- last_MAP %.4f -- last_MRR %.4f'% (
                best_map_val, best_mrr_val, last_dev_map, last_dev_mrr))
            print('--- best_MAP %.4f -- best_MRR %.4f -- last_MAP %.4f -- last_MRR %.4f' % (
                best_map_val, best_mrr_val, last_dev_map, last_dev_mrr))

def main():
    max_q_length = 25
    max_a_length = 90
    processed_data_path_pairwise = '../data/WikiQA/processed/pairwise'
    train_file = os.path.join(processed_data_path_pairwise, 'WikiQA-train-triplets.tsv')
    dev_file = os.path.join(processed_data_path_pairwise, 'WikiQA-dev.tsv')
    test_file = os.path.join(processed_data_path_pairwise, 'WikiQA-test.tsv')
    vocab = os.path.join(processed_data_path_pairwise, 'wiki_vocab.txt')

    config = NNConfig()
    config.ques_length = max_q_length
    config.ans_length = max_a_length

    da = Dataset(config)
    # da._genVocabFile(vocab)  # 生成vocabFile
    # da._genElmoEmbedding()  # 生成elmo_token_embedding

    train_transform = transform_train(train_file)
    dev_transform = transform(dev_file)
    test_transform = transform(test_file)
    train_corpus = load_train_data(train_transform, max_q_length, max_a_length)
    dev_corpus = load_data(dev_transform, max_q_length, max_a_length, keep_ids=True)
    test_corpus = load_data(test_transform, max_q_length, max_a_length, keep_ids=True)

    train(deepcopy(train_corpus), dev_corpus, test_corpus, config)

if __name__ == '__main__':
    save_path = "./model/checkpoint"
    best_path = "./model/bestval"
    main()

# def main(args):
#     max_q_length = 25
#     max_a_length = 90
#     processed_data_path_pairwise = '../data/WikiQA/processed/pairwise'
#     train_file = os.path.join(processed_data_path_pairwise, 'WikiQA-train-triplets.tsv')
#     dev_file = os.path.join(processed_data_path_pairwise, 'WikiQA-dev.tsv')
#     test_file = os.path.join(processed_data_path_pairwise, 'WikiQA-test.tsv')
#     vocab = os.path.join(processed_data_path_pairwise, 'wiki_clean_vocab.txt')
#     train_transform = transform_train(train_file, vocab)
#     dev_transform = transform(dev_file, vocab)
#     test_transform = transform(test_file, vocab)
#     train_corpus = load_train_data(train_transform, max_q_length, max_a_length)
#     dev_corpus = load_data(dev_transform, max_q_length, max_a_length, keep_ids=True)
#     test_corpus = load_data(test_transform, max_q_length, max_a_length, keep_ids=True)
#
#     config = NNConfig()
#     config.ques_length = max_q_length
#     config.ans_length = max_a_length
#     if args.train:
#         train(deepcopy(train_corpus), dev_corpus, test_corpus, config)
#     elif args.test:
#         test(test_corpus, config)
#
#
# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--train", help="whether to train", action='store_true')
#     parser.add_argument("--test", help="whether to test", action='store_true')
#     args = parser.parse_args()
#
#     save_path = "./model/checkpoint"
#     best_path = "./model/bestval"
#     main(args)
