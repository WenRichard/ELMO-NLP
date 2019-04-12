# -*- coding: utf-8 -*-
# @Time    : 2019/4/9 18:41
# @Author  : Alan
# @Email   : xiezhengwen2013@163.com
# @File    : train.py
# @Software: PyCharm

import os
import datetime
import numpy as np
import tensorflow as tf

from preprocrss import *
from model import BiLSTMAttention
from bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers, Batcher
from utils import *


class TrainingConfig(object):
    epoches = 10
    evaluateEvery = 100
    checkpointEvery = 100
    learningRate = 0.001


class ModelConfig(object):
    embeddingSize = 256  # 这个值是和ELMo模型的output Size 对应的值

    hiddenSizes = [128]  # LSTM结构的神经元个数

    dropoutKeepProb = 0.5
    l2RegLambda = 0.0


class Config(object):
    sequenceLength = 200  # 取了所有序列长度的均值
    batchSize = 128

    dataSource = "../data/preProcess/labeledTrain.csv"

    stopWordSource = "../data/english"

    optionFile = "modelParams/elmo_options.json"
    weightFile = "modelParams/elmo_weights.hdf5"
    vocabFile = "modelParams/vocab.txt"
    tokenEmbeddingFile = 'modelParams/elmo_token_embeddings.hdf5'

    numClasses = 2

    rate = 0.8  # 训练集的比例

    training = TrainingConfig()

    model = ModelConfig()


# 实例化配置参数对象
config = Config()
data = Dataset(config)
data.dataGen()

# 训练模型

# 生成训练集和验证集
trainReviews = data.trainReviews
# print([len(i) for i in trainReviews])
trainLabels = data.trainLabels
evalReviews = data.evalReviews
evalLabels = data.evalLabels

# 定义计算图

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth = True
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9  # 配置gpu占用率

    sess = tf.Session(config=session_conf)

    # 定义会话
    with sess.as_default():
        cnn = BiLSTMAttention(config)

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

        globalStep = tf.Variable(0, name="globalStep", trainable=False)
        # 定义优化函数，传入学习速率参数
        optimizer = tf.train.AdamOptimizer(config.training.learningRate)
        # 计算梯度,得到梯度和变量
        gradsAndVars = optimizer.compute_gradients(cnn.loss)
        # 将梯度应用到变量下，生成训练器
        trainOp = optimizer.apply_gradients(gradsAndVars, global_step=globalStep)

        # 用summary绘制tensorBoard
        gradSummaries = []
        for g, v in gradsAndVars:
            if g is not None:
                tf.summary.histogram("{}/grad/hist".format(v.name), g)
                tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))

        outDir = os.path.abspath(os.path.join(os.path.curdir, "summarys"))
        print("Writing to {}\n".format(outDir))

        lossSummary = tf.summary.scalar("loss", cnn.loss)
        summaryOp = tf.summary.merge_all()

        trainSummaryDir = os.path.join(outDir, "train")
        trainSummaryWriter = tf.summary.FileWriter(trainSummaryDir, sess.graph)

        evalSummaryDir = os.path.join(outDir, "eval")
        evalSummaryWriter = tf.summary.FileWriter(evalSummaryDir, sess.graph)

        # 初始化所有变量
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        # 保存模型的一种方式，保存为pb文件
        #         builder = tf.saved_model.builder.SavedModelBuilder("../model/textCNN/savedModel")
        sess.run(tf.global_variables_initializer())


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


        def trainStep(batchX, batchY):
            """
            训练函数
            """
            # print('s', np.shape(elmo(batchX)[0]))
            feed_dict = {
                cnn.inputX: elmo(batchX)[0],  # inputX直接用动态生成的ELMo向量表示代入
                cnn.inputY: np.array(batchY, dtype="float32"),
                cnn.dropoutKeepProb: config.model.dropoutKeepProb
            }
            _, summary, step, loss, predictions, binaryPreds = sess.run(
                [trainOp, summaryOp, globalStep, cnn.loss, cnn.predictions, cnn.binaryPreds],
                feed_dict)
            timeStr = datetime.datetime.now().isoformat()
            acc, auc, precision, recall = genMetrics(batchY, predictions, binaryPreds)
            print("{}, step: {}, loss: {}, acc: {}, auc: {}, precision: {}, recall: {}".format(timeStr, step, loss, acc,
                                                                                               auc, precision, recall))
            trainSummaryWriter.add_summary(summary, step)


        def devStep(batchX, batchY):
            """
            验证函数
            """
            feed_dict = {
                cnn.inputX: elmo(batchX)[0],
                cnn.inputY: np.array(batchY, dtype="float32"),
                cnn.dropoutKeepProb: 1.0
            }
            summary, step, loss, predictions, binaryPreds = sess.run(
                [summaryOp, globalStep, cnn.loss, cnn.predictions, cnn.binaryPreds],
                feed_dict)

            acc, auc, precision, recall = genMetrics(batchY, predictions, binaryPreds)

            evalSummaryWriter.add_summary(summary, step)

            return loss, acc, auc, precision, recall


        for i in range(config.training.epoches):
            # 训练模型
            print("start training model")
            for batchTrain in nextBatch(trainReviews, trainLabels, config.batchSize):
                trainStep(batchTrain[0], batchTrain[1])

                currentStep = tf.train.global_step(sess, globalStep)
                if currentStep % config.training.evaluateEvery == 0:
                    print("\nEvaluation:")

                    losses = []
                    accs = []
                    aucs = []
                    precisions = []
                    recalls = []

                    for batchEval in nextBatch(evalReviews, evalLabels, config.batchSize):
                        loss, acc, auc, precision, recall = devStep(batchEval[0], batchEval[1])
                        losses.append(loss)
                        accs.append(acc)
                        aucs.append(auc)
                        precisions.append(precision)
                        recalls.append(recall)

                    time_str = datetime.datetime.now().isoformat()
                    print("{}, step: {}, loss: {}, acc: {}, auc: {}, precision: {}, recall: {}".format(time_str,
                                                                                                       currentStep,
                                                                                                       mean(losses),
                                                                                                       mean(accs),
                                                                                                       mean(aucs),
                                                                                                       mean(precisions),
                                                                                                       mean(recalls)))

#                 if currentStep % config.training.checkpointEvery == 0:
#                     # 保存模型的另一种方法，保存checkpoint文件
#                     path = saver.save(sess, "../model/textCNN/model/my-model", global_step=currentStep)
#                     print("Saved model checkpoint to {}\n".format(path))

#         inputs = {"inputX": tf.saved_model.utils.build_tensor_info(cnn.inputX),
#                   "keepProb": tf.saved_model.utils.build_tensor_info(cnn.dropoutKeepProb)}

#         outputs = {"binaryPreds": tf.saved_model.utils.build_tensor_info(cnn.binaryPreds)}

#         prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs, outputs=outputs,
#                                                                                       method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
#         legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
#         builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
#                                             signature_def_map={"predict": prediction_signature}, legacy_init_op=legacy_init_op)

#         builder.save()